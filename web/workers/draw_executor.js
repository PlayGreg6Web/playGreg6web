'use strict';

// WebGPU draw executor with Frame-1 DrawInstanced(3, 1) / 24B-stride / CBV fallback profile
// routing (Track 3/4 task G38). Zero external dependencies; UMD export for Web Workers
// (globalThis.PlayGta6DrawExecutor) and Node.js (module.exports).
//
// Bridges missing console-native shaders to G34 fallback WGSL profiles in
// web/workers/shader_fallback_router.js (`fullscreen_triangle_cbv`, `textured_quad_24b`, `default`)
// while hardening bind-group and vertex-buffer bindings against unbound, destroyed, misaligned,
// or undersized Frame-1 bring-up resources.

(function (root) {
  const FALLBACK_PROFILE = Object.freeze({
    DEFAULT: 'default',
    FULLSCREEN_TRIANGLE_CBV: 'fullscreen_triangle_cbv',
    TEXTURED_QUAD_24B: 'textured_quad_24b',
    SPLASH_RECT_24B: 'splash_rect_24b',
  });

  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: FALLBACK_PROFILE.DEFAULT,
    minimal: FALLBACK_PROFILE.DEFAULT,
    fullscreen_triangle_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    fullscreen_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    'fullscreen-triangle-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    'fullscreen-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
    textured_quad_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    textured_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    'textured-quad-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    'textured-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
    splash_rect_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'splash-rect-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
    rectlist_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'rectlist-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
    frame1_splash_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
    'frame1-splash-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
  });

  const TEXTURED_QUAD_VERTEX_STRIDE = 24;
  const FRAME1_CBV_MIN_BYTES = 256;
  const UNIFORM_OFFSET_ALIGNMENT = 256;
  const VERTEX_OFFSET_ALIGNMENT = 4;
  const DEFAULT_COLOR_FORMAT = 'bgra8unorm';
  const DEFAULT_TOPOLOGY = 'triangle-list';
  const DEFAULT_MAX_FALLBACK_PIPELINES = 128;
  const DEFAULT_MAX_BIND_GROUP_CACHE_ENTRIES = 1024;
  const DEFAULT_MAX_WGSL_HASH_CACHE_ENTRIES = 256;
  const MAX_VIEWS_PER_TEXTURE = 64;
  const I32_MIN = -0x80000000;
  const I32_MAX = 0x7fffffff;
  const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;

  const PASSTHROUGH_DRAW_STATE_KEYS = Object.freeze([
    'hasBoundCbv',
    'hasCbv',
    'cbvCount',
    'cbv',
    'cbvBuffer',
    'uniformBuffer',
    'hasBoundTexture',
    'hasTexture',
    'hasSampledTexture',
    'hasBoundSampledTexture',
    'hasBoundSrv',
    'textureCount',
    'srvCount',
    'texture',
    'textureView',
    'srv',
    'hasBoundSampler',
    'hasSampler',
    'samplerCount',
    'sampler',
    'vertexShaderHash',
    'fragmentShaderHash',
    'vertexShader',
    'fragmentShader',
    'topology',
    'isRectList',
    'format',
    'sampleCount',
    'depthStencilFormat',
    'isFallback',
    'usesFallbackShaders',
  ]);

  // Fixed-slot fast paths for WebGPU guaranteed minimum slots (0..7 VBs, 0..3 BGs)
  // enabling O(1) bitmask dirty tracking and zero-allocation hot draw execution (W282).
  const MAX_FAST_VB_SLOTS = 8;
  const MAX_FAST_BG_SLOTS = 4;
  const MAX_FAST_DYN_OFFSETS = 8;

  // W354: O(1) zero-string-alloc hash-bucketed bind-group cache constants for 1..8 entries.
  const MAX_FAST_BG_CACHE_ENTRIES = 8;
  const BG_OWNER_LAYOUT = 1;
  const BG_OWNER_PIPELINE = 2;
  const BG_OWNER_FALLBACK_CBV = 3;
  const BG_OWNER_FALLBACK_TEX = 4;
  const BG_OWNER_FALLBACK_SPLASH = 5;
  const BG_ENTRY_KIND_BUFFER = 0;
  const BG_ENTRY_KIND_TEXTURE = 1;
  const BG_ENTRY_KIND_RAW = 2;

  function hashMixInt32(h, v) {
    let x = (h ^ (v | 0)) | 0;
    x = Math.imul(x, 0x01000193);
    return x | 0;
  }

  function hashMixNum(h, n) {
    if (n === 0) return hashMixInt32(h, 0);
    const lo = n | 0;
    if (lo === n) return hashMixInt32(h, lo);
    const hi = Math.floor(n / 4294967296) | 0;
    return hashMixInt32(hashMixInt32(h, lo), hi);
  }

  // Integer status codes for zero-allocation index/indirect buffer validation & pass binding.
  const IB_STATUS_INVALID = 0;
  const IB_STATUS_INVALID_RANGE = 1;
  const IB_STATUS_EMITTED = 2;
  const IB_STATUS_ELIDED = 3;

  const INDIRECT_STATUS_INVALID = 0;
  const INDIRECT_STATUS_INVALID_RANGE = 1;
  const INDIRECT_STATUS_OK = 2;

  // Bitmask flags cached per pipeline object for zero-allocation UI/glyph/video draw accounting.
  const PIPE_FLAG_ALPHA_BLEND = 1 << 0;
  const PIPE_FLAG_UI_QUAD = 1 << 1;
  const PIPE_FLAG_FONT_GLYPH = 1 << 2;
  const PIPE_FLAG_VIDEO = 1 << 3;
  const PIPE_FLAG_RECTLIST = 1 << 4;
  const PIPE_FLAG_STRIDE_24 = 1 << 5;
  const PIPE_FLAG_CLEAR = 1 << 6;

  const FALLBACK_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: TEXTURED_QUAD_VERTEX_STRIDE,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
      Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
    ]),
  });

  /**
   * Exact real Frame-1 24-byte splash pass vertex layout (G37 / G47):
   *   POSITION: R32G32B32_FLOAT (float32x3 @ 0, @location(0))
   *   COLOR:    R8G8B8A8_UNORM  (unorm8x4  @ 12, @location(1))
   *   TEXCOORD: R32G32_FLOAT    (float32x2 @ 16, @location(2))
   */
  const FRAME1_SPLASH_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: TEXTURED_QUAD_VERTEX_STRIDE,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x3' }),
      Object.freeze({ shaderLocation: 1, offset: 12, format: 'unorm8x4' }),
      Object.freeze({ shaderLocation: 2, offset: 16, format: 'float32x2' }),
    ]),
  });

  // WebGPU GPUBufferUsage / GPUTextureUsage bitmasks with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    INDEX:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDEX === 'number'
        ? GPUBufferUsage.INDEX
        : 0x0010,
    VERTEX:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.VERTEX === 'number'
        ? GPUBufferUsage.VERTEX
        : 0x0020,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
    INDIRECT:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDIRECT === 'number'
        ? GPUBufferUsage.INDIRECT
        : 0x0100,
  });

  const TEXTURE_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
  });

  // 64-bit FNV-1a constants matching web/workers/wgsl_pack_loader.js & wgpu_worker.js.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MASK = 0xffffffffffffffffn;

  // Minimal safe built-in WGSL used only when no ShaderFallbackRouter is available at all.
  const EMERGENCY_BUILTIN_WGSL = Object.freeze({
    vertex: [
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    let p = positions[vertex_index % 3u];',
      '    return vec4<f32>(p, 0.0, 1.0);',
      '}',
      '',
    ].join('\n'),
    fragment: [
      '@fragment',
      'fn fs_main() -> @location(0) vec4<f32> {',
      '    return vec4<f32>(0.05, 0.07, 0.11, 1.0);',
      '}',
      '',
    ].join('\n'),
  });

  function normalizeProfileName(profile) {
    if (typeof profile !== 'string') {
      return FALLBACK_PROFILE.DEFAULT;
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return FALLBACK_PROFILE.DEFAULT;
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return FALLBACK_PROFILE.DEFAULT;
  }

  function toNonNegativeInt(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return Math.trunc(value);
  }

  function fnv1a64Hex(text) {
    const str = typeof text === 'string' ? text : String(text || '');
    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < str.length; i += 1) {
      hash ^= BigInt(str.charCodeAt(i) & 0xff);
      hash = (hash * FNV1A_64_PRIME) & U64_MASK;
    }
    return `0x${hash.toString(16).padStart(16, '0')}`;
  }

  function extractEntryPoint(wgsl, stageName) {
    const defaultName = stageName === 'fragment' ? 'fs_main' : 'vs_main';
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      return defaultName;
    }
    const re = new RegExp(`@${stageName}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const match = re.exec(wgsl);
    if (match) {
      return match[1];
    }
    const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(wgsl);
    return fallback ? fallback[1] : defaultName;
  }

  function iterateCollectionValues(collection) {
    if (!collection) {
      return [];
    }
    if (collection instanceof Map) {
      return Array.from(collection.values());
    }
    if (Array.isArray(collection)) {
      return collection;
    }
    if (typeof collection === 'object') {
      return Object.values(collection);
    }
    return [];
  }

  function hasAnyTruthyValue(collection) {
    const values = iterateCollectionValues(collection);
    for (let i = 0; i < values.length; i += 1) {
      const item = values[i];
      if (item !== null && item !== undefined && item !== false) {
        if (typeof item === 'object') {
          if (item.destroyed === true) continue;
          if ('buffer' in item && item.buffer === null && item.present !== true) continue;
          if ('texture' in item && item.texture === null && !item.textureView && item.present !== true) continue;
          if ('sampler' in item && item.sampler === null && item.present !== true) continue;
        }
        return true;
      }
    }
    return false;
  }

  function inspectBindGroupMeta(bindGroups) {
    let hasCbv = false;
    let hasTexture = false;
    let hasSampler = false;

    const groups = iterateCollectionValues(bindGroups);
    for (let i = 0; i < groups.length; i += 1) {
      const group = groups[i];
      if (!group || typeof group !== 'object') {
        continue;
      }
      if (group.hasCbv === true || group.hasBoundCbv === true || group.cbv || group.cbvBuffer || group.uniformBuffer) {
        hasCbv = true;
      }
      if (
        group.hasTexture === true ||
        group.hasBoundTexture === true ||
        group.hasSampledTexture === true ||
        group.hasSrv === true ||
        group.texture ||
        group.textureView ||
        group.srv
      ) {
        hasTexture = true;
      }
      if (group.hasSampler === true || group.hasBoundSampler === true || group.sampler) {
        hasSampler = true;
      }

      const entries = Array.isArray(group.entries)
        ? group.entries
        : group.desc && Array.isArray(group.desc.entries)
          ? group.desc.entries
          : null;
      if (!entries) {
        continue;
      }
      for (let j = 0; j < entries.length; j += 1) {
        const entry = entries[j];
        if (!entry || typeof entry !== 'object') {
          continue;
        }
        const type = typeof entry.type === 'string' ? entry.type.toLowerCase() : '';
        const kind = typeof entry.kind === 'string' ? entry.kind.toLowerCase() : '';
        const resourceType = entry.resourceType;
        const resource = entry.resource;
        const resKind =
          resource && typeof resource === 'object' && typeof resource.kind === 'string'
            ? resource.kind.toLowerCase()
            : '';

        if (
          type === 'uniform' ||
          type === 'cbv' ||
          kind === 'cbv' ||
          kind === 'uniform' ||
          entry.isCbv === true ||
          resourceType === 0 ||
          resourceType === 'buffer' ||
          (resource && typeof resource === 'object' && resource.buffer)
        ) {
          hasCbv = true;
        }
        if (
          type === 'texture' ||
          type === 'srv' ||
          type === 'sampled-texture' ||
          kind === 'srv' ||
          kind === 'texture' ||
          kind === 'gputextureview' ||
          resKind === 'gputextureview' ||
          resKind === 'gputexture' ||
          entry.isTexture === true ||
          resourceType === 1 ||
          resourceType === 'textureView' ||
          resourceType === 'texture' ||
          entry.textureView ||
          entry.texture
        ) {
          hasTexture = true;
        }
        if (
          type === 'sampler' ||
          kind === 'sampler' ||
          kind === 'gpusampler' ||
          resKind === 'gpusampler' ||
          entry.isSampler === true ||
          resourceType === 2 ||
          resourceType === 'sampler' ||
          entry.sampler
        ) {
          hasSampler = true;
        }
      }
    }

    return { hasCbv, hasTexture, hasSampler };
  }

  function hasVertexStride(drawState, targetStride) {
    if (
      drawState.vertexStride === targetStride ||
      drawState.arrayStride === targetStride ||
      drawState.vertexBufferStride === targetStride
    ) {
      return true;
    }

    const vbValues = iterateCollectionValues(drawState.vertexBuffers);
    for (let i = 0; i < vbValues.length; i += 1) {
      const vb = vbValues[i];
      if (!vb || typeof vb !== 'object') {
        continue;
      }
      if (vb.stride === targetStride || vb.arrayStride === targetStride) {
        return true;
      }
    }

    const layoutValues = iterateCollectionValues(
      drawState.vertexBufferLayouts || drawState.vertexLayouts
    );
    for (let i = 0; i < layoutValues.length; i += 1) {
      const layout = layoutValues[i];
      if (!layout || typeof layout !== 'object') {
        continue;
      }
      if (layout.arrayStride === targetStride || layout.stride === targetStride) {
        return true;
      }
    }

    return false;
  }

  function isRectListTopology(topology) {
    if (typeof topology !== 'string') {
      return false;
    }
    const norm = topology.trim().toLowerCase();
    return (
      norm === 'rect-list' ||
      norm === 'rectlist' ||
      norm === 'd3d_primitive_topology_rectlist'
    );
  }

  function isRectListDrawState(drawState) {
    if (!drawState || typeof drawState !== 'object') {
      return false;
    }
    if (drawState.isRectList === true || isRectListTopology(drawState.topology)) {
      return true;
    }
    const pipe =
      drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
    if (pipe && (pipe.isRectList === true || isRectListTopology(pipe.topology))) {
      return true;
    }
    return false;
  }

  /**
   * Expand Xbox `D3D_PRIMITIVE_TOPOLOGY_RECTLIST` 24-byte vertices (3 vertices per quad:
   * `v0, v1, v2`) into WebGPU `triangle-list` vertices (6 vertices per quad:
   * `[v0, v1, v2, v1, v3, v2]`).
   *
   * Frame-1 24-byte vertex layout:
   *   bytes  0..11: POSITION (`float32x3`, little-endian)
   *   bytes 12..15: COLOR    (`unorm8x4`)
   *   bytes 16..23: TEXCOORD (`float32x2`, little-endian)
   *
   * Implicit 4th corner `v3`:
   *   `pos3 = pos1 + pos2 - pos0`
   *   `color3 = color2` (bytes 12..15 copied verbatim)
   *   `uv3 = uv1 + uv2 - uv0`
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} srcBytes
   * @param {number} [vertexCount]
   * @returns {Uint8Array}
   */
  function expandRectListVertices24B(srcBytes, vertexCount) {
    let bytes;
    if (srcBytes instanceof Uint8Array) {
      bytes = srcBytes;
    } else if (ArrayBuffer.isView(srcBytes)) {
      bytes = new Uint8Array(srcBytes.buffer, srcBytes.byteOffset, srcBytes.byteLength);
    } else if (srcBytes instanceof ArrayBuffer) {
      bytes = new Uint8Array(srcBytes);
    } else {
      return new Uint8Array(0);
    }

    const stride = TEXTURED_QUAD_VERTEX_STRIDE;
    const maxAvailableVerts = Math.floor(bytes.byteLength / stride);
    const effectiveVerts =
      vertexCount === undefined || vertexCount === null
        ? maxAvailableVerts
        : Math.min(toNonNegativeInt(vertexCount, 0), maxAvailableVerts);

    const rectCount = Math.floor(effectiveVerts / 3);
    if (rectCount <= 0) {
      return new Uint8Array(0);
    }

    const out = new Uint8Array(rectCount * 6 * stride);
    const srcView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const dstView = new DataView(out.buffer, out.byteOffset, out.byteLength);

    for (let r = 0; r < rectCount; r += 1) {
      const v0Off = r * 3 * stride;
      const v1Off = v0Off + stride;
      const v2Off = v1Off + stride;
      const dstBase = r * 6 * stride;

      // Triangle 1: [v0, v1, v2]
      out.set(bytes.subarray(v0Off, v0Off + stride), dstBase);
      out.set(bytes.subarray(v1Off, v1Off + stride), dstBase + stride);
      out.set(bytes.subarray(v2Off, v2Off + stride), dstBase + 2 * stride);

      // Triangle 2: [v1, v3, v2]
      out.set(bytes.subarray(v1Off, v1Off + stride), dstBase + 3 * stride);

      const v3DstOff = dstBase + 4 * stride;
      const x3 =
        srcView.getFloat32(v1Off + 0, true) +
        srcView.getFloat32(v2Off + 0, true) -
        srcView.getFloat32(v0Off + 0, true);
      const y3 =
        srcView.getFloat32(v1Off + 4, true) +
        srcView.getFloat32(v2Off + 4, true) -
        srcView.getFloat32(v0Off + 4, true);
      const z3 =
        srcView.getFloat32(v1Off + 8, true) +
        srcView.getFloat32(v2Off + 8, true) -
        srcView.getFloat32(v0Off + 8, true);
      dstView.setFloat32(v3DstOff + 0, x3, true);
      dstView.setFloat32(v3DstOff + 4, y3, true);
      dstView.setFloat32(v3DstOff + 8, z3, true);

      // color3 = color2 (unorm8x4 bytes 12..15)
      out[v3DstOff + 12] = bytes[v2Off + 12];
      out[v3DstOff + 13] = bytes[v2Off + 13];
      out[v3DstOff + 14] = bytes[v2Off + 14];
      out[v3DstOff + 15] = bytes[v2Off + 15];

      const u3 =
        srcView.getFloat32(v1Off + 16, true) +
        srcView.getFloat32(v2Off + 16, true) -
        srcView.getFloat32(v0Off + 16, true);
      const v3 =
        srcView.getFloat32(v1Off + 20, true) +
        srcView.getFloat32(v2Off + 20, true) -
        srcView.getFloat32(v0Off + 20, true);
      dstView.setFloat32(v3DstOff + 16, u3, true);
      dstView.setFloat32(v3DstOff + 20, v3, true);

      out.set(bytes.subarray(v2Off, v2Off + stride), dstBase + 5 * stride);
    }

    return out;
  }

  /**
   * Pure helper that selects the appropriate G34/G47 built-in fallback WGSL profile from active
   * draw state when real shaders are missing or fallback-routed:
   * - Xbox `rect-list` topology (`topology === 'rect-list'` or `isRectList === true`) with
   *   24-byte-stride vertex layout -> `'splash_rect_24b'`
   * - `vertexCount === 3` with a bound CBV and no bound sampled texture (real title Frame-1
   *   `DrawInstanced(3, 1)` pattern) -> `'fullscreen_triangle_cbv'`
   * - 24-byte-stride vertex buffer with a bound texture/sampler -> `'textured_quad_24b'`
   * - Otherwise -> `'default'`
   *
   * @param {object | null | undefined} drawState
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
   */
  function selectFallbackShaderProfile(drawState) {
    if (!drawState || typeof drawState !== 'object') {
      return FALLBACK_PROFILE.DEFAULT;
    }

    const vertexCount = toNonNegativeInt(drawState.vertexCount, 0);
    const has24BStride = hasVertexStride(drawState, TEXTURED_QUAD_VERTEX_STRIDE);

    // 0. Xbox RECTLIST draw with Frame-1 24-byte vertex stride (POSITION + COLOR + TEXCOORD).
    if (isRectListDrawState(drawState) && has24BStride) {
      return FALLBACK_PROFILE.SPLASH_RECT_24B;
    }

    const bgInfo = inspectBindGroupMeta(drawState.bindGroups);

    const hasBoundCbv =
      drawState.hasBoundCbv === true ||
      drawState.hasCbv === true ||
      toNonNegativeInt(drawState.cbvCount, 0) > 0 ||
      Boolean(drawState.cbv || drawState.cbvBuffer || drawState.uniformBuffer) ||
      hasAnyTruthyValue(drawState.cbvs) ||
      hasAnyTruthyValue(drawState.uniformBuffers) ||
      bgInfo.hasCbv;

    const hasBoundTexture =
      drawState.hasBoundTexture === true ||
      drawState.hasTexture === true ||
      drawState.hasSampledTexture === true ||
      drawState.hasBoundSampledTexture === true ||
      drawState.hasBoundSrv === true ||
      toNonNegativeInt(drawState.textureCount, 0) > 0 ||
      toNonNegativeInt(drawState.srvCount, 0) > 0 ||
      Boolean(drawState.texture || drawState.textureView || drawState.srv) ||
      hasAnyTruthyValue(drawState.textures) ||
      hasAnyTruthyValue(drawState.textureViews) ||
      hasAnyTruthyValue(drawState.srvs) ||
      bgInfo.hasTexture;

    const hasBoundSampler =
      drawState.hasBoundSampler === true ||
      drawState.hasSampler === true ||
      toNonNegativeInt(drawState.samplerCount, 0) > 0 ||
      Boolean(drawState.sampler) ||
      hasAnyTruthyValue(drawState.samplers) ||
      bgInfo.hasSampler;

    // 1. Frame-1 DrawInstanced(3, 1) with bound CBV and no sampled texture.
    if (vertexCount === 3 && hasBoundCbv && !hasBoundTexture) {
      return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
    }

    // 2. 24-byte-stride vertex buffer with bound texture or sampler.
    if (has24BStride && (hasBoundTexture || hasBoundSampler)) {
      return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
    }

    return FALLBACK_PROFILE.DEFAULT;
  }

  function resolveIndirectDrawModule() {
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6IndirectDraw &&
      typeof globalThis.PlayGta6IndirectDraw.GpuIndirectClampPass === 'function'
    ) {
      return globalThis.PlayGta6IndirectDraw;
    }
    if (typeof importScripts === 'function') {
      try {
        importScripts('./indirect_draw.js');
      } catch (_err) {
        // Optional helper unavailable in this worker context.
      }
      if (
        typeof globalThis !== 'undefined' &&
        globalThis.PlayGta6IndirectDraw &&
        typeof globalThis.PlayGta6IndirectDraw.GpuIndirectClampPass === 'function'
      ) {
        return globalThis.PlayGta6IndirectDraw;
      }
    }
    if (typeof require === 'function') {
      try {
        const mod = require('./indirect_draw.js');
        if (mod && typeof mod.GpuIndirectClampPass === 'function') {
          return mod;
        }
      } catch (_err) {
        // Optional helper unavailable in this Node context.
      }
    }
    return null;
  }

  function normalizeSignedBaseVertex(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return 0;
    }
    const t = Math.trunc(value);
    if (t < I32_MIN) return I32_MIN;
    if (t > I32_MAX) return I32_MAX;
    return t | 0;
  }

  function resolveShaderFallbackRouterCtor() {
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6ShaderFallbackRouter &&
      typeof globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter === 'function'
    ) {
      return globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter;
    }
    if (typeof importScripts === 'function') {
      try {
        importScripts('./shader_fallback_router.js');
      } catch (_err) {
        // Optional helper unavailable in this worker context.
      }
      if (
        typeof globalThis !== 'undefined' &&
        globalThis.PlayGta6ShaderFallbackRouter &&
        typeof globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter === 'function'
      ) {
        return globalThis.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter;
      }
    }
    if (typeof require === 'function') {
      try {
        const mod = require('./shader_fallback_router.js');
        if (mod && typeof mod.ShaderFallbackRouter === 'function') {
          return mod.ShaderFallbackRouter;
        }
      } catch (_err) {
        // Optional helper unavailable in this Node context.
      }
    }
    return null;
  }

  function unwrapRouterResult(raw, stageName) {
    if (raw === null || raw === undefined) {
      return null;
    }
    if (typeof raw === 'string' && raw.length > 0) {
      return {
        wgsl: raw,
        source: 'builtin',
        entryPoint: extractEntryPoint(raw, stageName),
        resolvedHash: null,
        profile: null,
      };
    }
    if (typeof raw === 'object') {
      const wgsl =
        typeof raw.wgsl === 'string'
          ? raw.wgsl
          : typeof raw.code === 'string'
            ? raw.code
            : typeof raw.source === 'string' && raw.source.includes('fn ')
              ? raw.source
              : null;
      if (!wgsl || wgsl.length === 0) {
        return null;
      }
      const source =
        typeof raw.source === 'string' && !raw.source.includes('fn ') ? raw.source : 'builtin';
      const entryPoint =
        typeof raw.entryPoint === 'string' && raw.entryPoint.length > 0
          ? raw.entryPoint
          : extractEntryPoint(wgsl, stageName);
      const profile =
        typeof raw.profile === 'string' && raw.profile.length > 0
          ? normalizeProfileName(raw.profile)
          : null;
      return {
        wgsl,
        source,
        entryPoint,
        resolvedHash: raw.resolvedHash !== undefined ? raw.resolvedHash : null,
        profile,
      };
    }
    return null;
  }

  function inferEffectiveProfileFromWgsl(vsResult, fsResult) {
    const vsProfile = vsResult && vsResult.profile ? vsResult.profile : null;
    const fsProfile = fsResult && fsResult.profile ? fsResult.profile : null;
    if (vsProfile && fsProfile && vsProfile === fsProfile) {
      return vsProfile;
    }

    const vsCode = vsResult && typeof vsResult.wgsl === 'string' ? vsResult.wgsl : '';
    const fsCode = fsResult && typeof fsResult.wgsl === 'string' ? fsResult.wgsl : '';

    if (
      /@location\(0\)[\s\S]*?vec3<f32>[\s\S]*?@location\(1\)[\s\S]*?vec4<f32>[\s\S]*?@location\(2\)[\s\S]*?vec2<f32>/.test(
        vsCode
      ) &&
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(1\)[\s\S]*?texture_2d<f32>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(2\)[\s\S]*?sampler/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.SPLASH_RECT_24B;
    }
    if (
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(vsCode) ||
      /@group\(0\)\s+@binding\(0\)\s+var<uniform>/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
    }
    if (
      /@location\(0\)[\s\S]*?vec4<f32>[\s\S]*?@location\(1\)[\s\S]*?vec2<f32>/.test(vsCode) &&
      /@group\(0\)\s+@binding\(0\)[\s\S]*?texture_2d<f32>/.test(fsCode) &&
      /@group\(0\)\s+@binding\(1\)[\s\S]*?sampler/.test(fsCode)
    ) {
      return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
    }

    return vsProfile || fsProfile || FALLBACK_PROFILE.DEFAULT;
  }

  function isValidGpuBuffer(buffer, requiredUsage) {
    if (!buffer || typeof buffer !== 'object') {
      return false;
    }
    if (buffer.destroyed === true) {
      return false;
    }
    if (
      typeof requiredUsage === 'number' &&
      requiredUsage !== 0 &&
      typeof buffer.usage === 'number' &&
      (buffer.usage & requiredUsage) === 0
    ) {
      return false;
    }
    return true;
  }

  function validateIndirectBufferStatus(indirectBuffer, indirectOffset, indexed) {
    if (!isValidGpuBuffer(indirectBuffer, BUFFER_USAGE.INDIRECT)) {
      return INDIRECT_STATUS_INVALID;
    }
    if (!Number.isInteger(indirectOffset) || indirectOffset < 0 || indirectOffset % 4 !== 0) {
      return INDIRECT_STATUS_INVALID_RANGE;
    }
    const minBytes = indexed ? 20 : 16;
    const bufSize =
      typeof indirectBuffer.size === 'number' && Number.isFinite(indirectBuffer.size)
        ? indirectBuffer.size
        : null;
    if (bufSize !== null) {
      if (bufSize < minBytes || indirectOffset + minBytes > bufSize) {
        return INDIRECT_STATUS_INVALID_RANGE;
      }
    }
    return INDIRECT_STATUS_OK;
  }

  function isValidIndirectBuffer(indirectBuffer, indirectOffset, indexed) {
    return validateIndirectBufferStatus(indirectBuffer, indirectOffset, indexed) === INDIRECT_STATUS_OK;
  }

  function isValidCbvBinding(entry) {
    if (!entry) {
      return false;
    }
    const buffer = entry.buffer !== undefined ? entry.buffer : entry;
    if (!isValidGpuBuffer(buffer, BUFFER_USAGE.UNIFORM)) {
      return false;
    }
    const offset = entry.offset !== undefined ? entry.offset : 0;
    if (!Number.isInteger(offset) || offset < 0 || offset % UNIFORM_OFFSET_ALIGNMENT !== 0) {
      return false;
    }
    const bufSize = typeof buffer.size === 'number' && Number.isFinite(buffer.size) ? buffer.size : null;
    const explicitSize =
      entry.size !== undefined && entry.size !== null
        ? entry.size
        : bufSize !== null
          ? bufSize - offset
          : null;
    if (explicitSize !== null) {
      if (!Number.isFinite(explicitSize) || explicitSize < FRAME1_CBV_MIN_BYTES) {
        return false;
      }
      if (bufSize !== null && offset + explicitSize > bufSize) {
        return false;
      }
    }
    return true;
  }

  function isValidVertexBufferParts(buffer, offset, size, stride, requiredBytes, expectedStride) {
    if (!isValidGpuBuffer(buffer, BUFFER_USAGE.VERTEX)) {
      return false;
    }
    const normOffset = offset !== undefined ? offset : 0;
    if (
      !Number.isInteger(normOffset) ||
      normOffset < 0 ||
      normOffset % VERTEX_OFFSET_ALIGNMENT !== 0
    ) {
      return false;
    }
    if (
      expectedStride !== undefined &&
      expectedStride !== null &&
      stride !== undefined &&
      stride !== null
    ) {
      if (!Number.isFinite(stride) || stride <= 0 || stride !== expectedStride) {
        return false;
      }
    }
    const bufSize =
      typeof buffer.size === 'number' && Number.isFinite(buffer.size) ? buffer.size : null;
    if (bufSize !== null) {
      if (bufSize <= 0 || normOffset >= bufSize) {
        return false;
      }
      const explicitSize = size !== undefined && size !== null ? size : bufSize - normOffset;
      if (!Number.isFinite(explicitSize) || explicitSize <= 0) {
        return false;
      }
      if (normOffset + explicitSize > bufSize) {
        return false;
      }
      if (requiredBytes > 0 && explicitSize < requiredBytes) {
        return false;
      }
    }
    return true;
  }

  function isValidVertexBufferBinding(entry, requiredBytes, expectedStride) {
    if (!entry) {
      return false;
    }
    const buffer = entry.buffer !== undefined ? entry.buffer : entry;
    return isValidVertexBufferParts(
      buffer,
      entry.offset,
      entry.size,
      entry.stride,
      requiredBytes,
      expectedStride
    );
  }

  function isValidTextureOrView(resource) {
    if (!resource || typeof resource !== 'object') {
      return false;
    }
    if (resource.destroyed === true) {
      return false;
    }
    return true;
  }

  function resolveTextureView(resource) {
    if (!isValidTextureOrView(resource)) {
      return null;
    }
    if (resource.textureView && isValidTextureOrView(resource.textureView)) {
      return resource.textureView;
    }
    if (resource.texture && isValidTextureOrView(resource.texture)) {
      if (typeof resource.texture.createView === 'function') {
        try {
          return resource.texture.createView();
        } catch (_err) {
          return null;
        }
      }
    }
    if (typeof resource.createView === 'function') {
      try {
        return resource.createView();
      } catch (_err) {
        return null;
      }
    }
    // Already a GPUTextureView (no createView method).
    return resource;
  }

  function isValidSampler(resource) {
    if (!resource || typeof resource !== 'object') {
      return false;
    }
    if (resource.destroyed === true) {
      return false;
    }
    if (resource.sampler && typeof resource.sampler === 'object') {
      return resource.sampler.destroyed !== true;
    }
    return true;
  }

  function unwrapSampler(resource) {
    if (!isValidSampler(resource)) {
      return null;
    }
    return resource.sampler && typeof resource.sampler === 'object' ? resource.sampler : resource;
  }

  function dynamicOffsetsEqual(boundRecord, nextOffsets) {
    const nextHas = nextOffsets !== null && nextOffsets !== undefined;
    if (boundRecord.hasDynamicOffsets !== nextHas) {
      return false;
    }
    if (!nextHas) {
      return true;
    }
    const nextLen = nextOffsets.length;
    if (boundRecord.dynamicOffsetsLength !== nextLen) {
      return false;
    }
    const prev = boundRecord.dynamicOffsetsSnapshot;
    for (let i = 0; i < nextLen; i += 1) {
      if (prev[i] !== nextOffsets[i]) {
        return false;
      }
    }
    return true;
  }

  function copyDynamicOffsetsInto(boundRecord, nextOffsets) {
    const nextHas = nextOffsets !== null && nextOffsets !== undefined;
    boundRecord.hasDynamicOffsets = nextHas;
    if (!nextHas) {
      boundRecord.dynamicOffsetsLength = 0;
      return;
    }
    const len = nextOffsets.length;
    boundRecord.dynamicOffsetsLength = len;
    if (len === 0) {
      return;
    }
    let snap = boundRecord.dynamicOffsetsSnapshot;
    if (!snap) {
      snap = new Array(len);
      boundRecord.dynamicOffsetsSnapshot = snap;
    }
    for (let i = 0; i < len; i += 1) {
      snap[i] = nextOffsets[i];
    }
  }

  function getResourceGeneration(resource) {
    if (resource && typeof resource === 'object' && typeof resource.generation === 'number') {
      return resource.generation;
    }
    return 0;
  }

  function buildTextureViewDescriptorKey(viewDesc) {
    if (!viewDesc || typeof viewDesc !== 'object') {
      return '';
    }
    const format = viewDesc.format !== undefined && viewDesc.format !== null ? viewDesc.format : '';
    const dimension =
      viewDesc.dimension !== undefined && viewDesc.dimension !== null ? viewDesc.dimension : '';
    const aspect = viewDesc.aspect !== undefined && viewDesc.aspect !== null ? viewDesc.aspect : '';
    const baseMip =
      viewDesc.baseMipLevel !== undefined && viewDesc.baseMipLevel !== null
        ? viewDesc.baseMipLevel
        : '';
    const mipCount =
      viewDesc.mipLevelCount !== undefined && viewDesc.mipLevelCount !== null
        ? viewDesc.mipLevelCount
        : '';
    const baseLayer =
      viewDesc.baseArrayLayer !== undefined && viewDesc.baseArrayLayer !== null
        ? viewDesc.baseArrayLayer
        : '';
    const layerCount =
      viewDesc.arrayLayerCount !== undefined && viewDesc.arrayLayerCount !== null
        ? viewDesc.arrayLayerCount
        : '';
    const usage = viewDesc.usage !== undefined && viewDesc.usage !== null ? viewDesc.usage : '';
    if (
      format === '' &&
      dimension === '' &&
      aspect === '' &&
      baseMip === '' &&
      mipCount === '' &&
      baseLayer === '' &&
      layerCount === '' &&
      usage === ''
    ) {
      return '';
    }
    return `${format}|${dimension}|${aspect}|${baseMip}|${mipCount}|${baseLayer}|${layerCount}|${usage}`;
  }

  function validateComputeIndirectBufferStatus(indirectBuffer, indirectOffset) {
    if (!isValidGpuBuffer(indirectBuffer, BUFFER_USAGE.INDIRECT)) {
      return INDIRECT_STATUS_INVALID;
    }
    if (!Number.isInteger(indirectOffset) || indirectOffset < 0 || indirectOffset % 4 !== 0) {
      return INDIRECT_STATUS_INVALID_RANGE;
    }
    const minBytes = 12; // 3 x u32 workgroup counts
    const bufSize =
      typeof indirectBuffer.size === 'number' && Number.isFinite(indirectBuffer.size)
        ? indirectBuffer.size
        : null;
    if (bufSize !== null) {
      if (bufSize < minBytes || indirectOffset + minBytes > bufSize) {
        return INDIRECT_STATUS_INVALID_RANGE;
      }
    }
    return INDIRECT_STATUS_OK;
  }

  function isCachedBindGroupValid(entry) {
    if (!entry || !entry.bindGroup || entry.bindGroup.destroyed === true) {
      return false;
    }
    const resources = entry.resources;
    if (resources) {
      const gens = entry.generations;
      for (let i = 0; i < resources.length; i += 1) {
        const res = resources[i];
        if (!res) {
          continue;
        }
        if (res.destroyed === true) {
          return false;
        }
        if (res.texture && res.texture.destroyed === true) {
          return false;
        }
        if (gens !== null && gens !== undefined && getResourceGeneration(res) !== gens[i]) {
          return false;
        }
      }
    }
    return true;
  }

  function hasAlphaBlendDescriptor(pipe) {
    if (!pipe || typeof pipe !== 'object') {
      return false;
    }
    if (pipe.alphaBlended === true || pipe.blendEnabled === true) {
      return true;
    }
    if (pipe.blend && typeof pipe.blend === 'object') {
      return true;
    }
    const desc = pipe.desc;
    if (desc && desc.fragment && Array.isArray(desc.fragment.targets)) {
      const targets = desc.fragment.targets;
      for (let i = 0; i < targets.length; i += 1) {
        const t = targets[i];
        if (t && t.blend && typeof t.blend === 'object') {
          return true;
        }
      }
    }
    return false;
  }

  // Status codes for zero-allocation pass state binding helpers.
  const BIND_STATUS_UNSUPPORTED = -1;
  const BIND_STATUS_ELIDED = 0;
  const BIND_STATUS_EMITTED = 1;

  function isRenderPassObject(value, methodName) {
    return (
      value !== null &&
      typeof value === 'object' &&
      typeof value[methodName] === 'function'
    );
  }

  function normalizeViewportArgsInto(out, xOrDesc, y, width, height, minDepth, maxDepth) {
    let vx;
    let vy;
    let vw;
    let vh;
    let vMin;
    let vMax;
    if (xOrDesc && typeof xOrDesc === 'object') {
      vx = xOrDesc.x;
      vy = xOrDesc.y;
      vw = xOrDesc.width;
      vh = xOrDesc.height;
      vMin = xOrDesc.minDepth !== undefined ? xOrDesc.minDepth : 0;
      vMax = xOrDesc.maxDepth !== undefined ? xOrDesc.maxDepth : 1;
    } else {
      vx = xOrDesc;
      vy = y;
      vw = width;
      vh = height;
      vMin = minDepth !== undefined ? minDepth : 0;
      vMax = maxDepth !== undefined ? maxDepth : 1;
    }
    if (
      typeof vx !== 'number' ||
      !Number.isFinite(vx) ||
      typeof vy !== 'number' ||
      !Number.isFinite(vy) ||
      typeof vw !== 'number' ||
      !Number.isFinite(vw) ||
      vw < 0 ||
      typeof vh !== 'number' ||
      !Number.isFinite(vh) ||
      vh < 0 ||
      typeof vMin !== 'number' ||
      !Number.isFinite(vMin) ||
      vMin < 0 ||
      vMin > 1 ||
      typeof vMax !== 'number' ||
      !Number.isFinite(vMax) ||
      vMax < 0 ||
      vMax > 1 ||
      vMin > vMax
    ) {
      return false;
    }
    out.x = vx;
    out.y = vy;
    out.width = vw;
    out.height = vh;
    out.minDepth = vMin;
    out.maxDepth = vMax;
    return true;
  }

  function normalizeScissorRectArgsInto(out, xOrDesc, y, width, height) {
    let sx;
    let sy;
    let sw;
    let sh;
    if (xOrDesc && typeof xOrDesc === 'object') {
      sx = xOrDesc.x;
      sy = xOrDesc.y;
      sw = xOrDesc.width;
      sh = xOrDesc.height;
    } else {
      sx = xOrDesc;
      sy = y;
      sw = width;
      sh = height;
    }
    if (
      !Number.isInteger(sx) ||
      sx < 0 ||
      !Number.isInteger(sy) ||
      sy < 0 ||
      !Number.isInteger(sw) ||
      sw < 0 ||
      !Number.isInteger(sh) ||
      sh < 0
    ) {
      return false;
    }
    out.x = sx;
    out.y = sy;
    out.width = sw;
    out.height = sh;
    return true;
  }

  function normalizeBlendConstantArgsInto(out, colorOrR, g, b, a) {
    let cr;
    let cg;
    let cb;
    let ca;
    if (Array.isArray(colorOrR) || ArrayBuffer.isView(colorOrR)) {
      if (colorOrR.length < 4) {
        return false;
      }
      cr = colorOrR[0];
      cg = colorOrR[1];
      cb = colorOrR[2];
      ca = colorOrR[3];
    } else if (colorOrR && typeof colorOrR === 'object') {
      cr = colorOrR.r;
      cg = colorOrR.g;
      cb = colorOrR.b;
      ca = colorOrR.a;
    } else {
      cr = colorOrR;
      cg = g;
      cb = b;
      ca = a;
    }
    if (
      typeof cr !== 'number' ||
      !Number.isFinite(cr) ||
      typeof cg !== 'number' ||
      !Number.isFinite(cg) ||
      typeof cb !== 'number' ||
      !Number.isFinite(cb) ||
      typeof ca !== 'number' ||
      !Number.isFinite(ca)
    ) {
      return false;
    }
    out.r = cr;
    out.g = cg;
    out.b = cb;
    out.a = ca;
    return true;
  }

  function normalizeStencilReference(reference) {
    if (!Number.isInteger(reference) || reference < 0 || reference > 0xffffffff) {
      return null;
    }
    return reference;
  }

  function createDrawResultObject(indexed, indirect, count, instanceCount, indirectOffset) {
    if (indirect) {
      return {
        executed: false,
        skipped: false,
        reason: null,
        isFallback: false,
        profile: null,
        effectiveProfile: null,
        indexed,
        indirect: true,
        indirectOffset,
        pipelineElided: false,
        bindGroupsElided: 0,
        vertexBuffersElided: 0,
        indexBufferElided: false,
        viewportElided: false,
        scissorElided: false,
        blendConstantElided: false,
        stencilReferenceElided: false,
        bindGroupCacheHit: false,
        vertexCount: 0,
        indexCount: 0,
        instanceCount: 0,
        substitutedCbv: false,
        substitutedTexture: false,
        substitutedSampler: false,
        substitutedVertexBuffer: false,
      };
    }
    return {
      executed: false,
      skipped: false,
      reason: null,
      isFallback: false,
      profile: null,
      effectiveProfile: null,
      indexed,
      indirect: false,
      pipelineElided: false,
      bindGroupsElided: 0,
      vertexBuffersElided: 0,
      indexBufferElided: false,
      viewportElided: false,
      scissorElided: false,
      blendConstantElided: false,
      stencilReferenceElided: false,
      bindGroupCacheHit: false,
      vertexCount: indexed ? 0 : count,
      indexCount: indexed ? count : 0,
      instanceCount,
      substitutedCbv: false,
      substitutedTexture: false,
      substitutedSampler: false,
      substitutedVertexBuffer: false,
    };
  }

  function resetDrawResultInPlace(out, indexed, indirect, count, instanceCount, indirectOffset) {
    out.executed = false;
    out.skipped = false;
    out.reason = null;
    out.isFallback = false;
    out.profile = null;
    out.effectiveProfile = null;
    out.indexed = indexed;
    out.indirect = indirect;
    if (indirect) {
      out.indirectOffset = indirectOffset;
    }
    out.pipelineElided = false;
    out.bindGroupsElided = 0;
    out.vertexBuffersElided = 0;
    out.indexBufferElided = false;
    out.viewportElided = false;
    out.scissorElided = false;
    out.blendConstantElided = false;
    out.stencilReferenceElided = false;
    out.bindGroupCacheHit = false;
    out.vertexCount = indirect || indexed ? 0 : count;
    out.indexCount = indirect || !indexed ? 0 : count;
    out.instanceCount = indirect ? 0 : instanceCount;
    out.substitutedCbv = false;
    out.substitutedTexture = false;
    out.substitutedSampler = false;
    out.substitutedVertexBuffer = false;
    return out;
  }

  function createZeroedStats() {
    return {
      totalDraws: 0,
      totalDrawsExecuted: 0,
      normalDraws: 0,
      fallbackDraws: 0,
      indexedDraws: 0,
      nonIndexedDraws: 0,
      indirectDraws: 0,
      skippedDraws: 0,
      zeroCountDrawsSkipped: 0,
      skippedNoPipeline: 0,
      skippedNoPass: 0,
      skippedInvalidRange: 0,
      skippedInvalidIndexBuffer: 0,
      skippedInvalidIndirectBuffer: 0,
      skippedBindFailed: 0,
      skippedPassError: 0,
      skippedDestroyed: 0,
      skippedByReason: {},
      redundantPipelineSkips: 0,
      redundantBindGroupSkips: 0,
      redundantVertexBufferSkips: 0,
      redundantIndexBufferSkips: 0,
      elidedPipelineSets: 0,
      elidedBindGroupSets: 0,
      elidedVertexBufferSets: 0,
      elidedIndexBufferSets: 0,
      elidedViewportScissorSets: 0,
      elidedBlendConstantSets: 0,
      elidedStencilReferenceSets: 0,
      totalElidedStateCalls: 0,
      bindGroupCacheHits: 0,
      bindGroupCacheMisses: 0,
      bindGroupCacheEvictions: 0,
      textureViewCacheHits: 0,
      textureViewCacheMisses: 0,
      textureViewCacheEvictions: 0,
      computeDispatches: 0,
      skippedComputeDispatches: 0,
      alphaBlendedDraws: 0,
      uiQuadDraws: 0,
      fontGlyphDraws: 0,
      videoDraws: 0,
      fallbackProfileCounts: {
        [FALLBACK_PROFILE.DEFAULT]: 0,
        [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]: 0,
        [FALLBACK_PROFILE.TEXTURED_QUAD_24B]: 0,
        [FALLBACK_PROFILE.SPLASH_RECT_24B]: 0,
      },
      fallbackPipelineCacheHits: 0,
      fallbackPipelineCacheMisses: 0,
      bindGroupGuards: 0,
      invalidCbvSubstitutions: 0,
      invalidTextureSubstitutions: 0,
      invalidSamplerSubstitutions: 0,
      skippedBindGroupBindings: 0,
      invalidVertexBufferGuards: 0,
      vertexBufferSubstitutions: 0,
      skippedVertexBufferBindings: 0,
      pipelineErrors: 0,
      bindGroupErrors: 0,
      passErrors: 0,
      legacyRouterFallbacks: 0,
      clampedIndirectDraws: 0,
      clampedIndirectDispatches: 0,
      indirectClampPasses: 0,
      indirectFirstInstanceClamped: 0,
      meshComputeDraws: 0,
      meshIndirectComputeDraws: 0,
      meshZeroCountElided: 0,
      multiPassInstancedDraws: 0,
      multiPassInstancedSubDraws: 0,
    };
  }

  /**
   * WebGPU draw executor with G34 fallback WGSL profile routing and Frame-1
   * bind-group / vertex-buffer hardening.
   */
  class DrawExecutor {
    /**
     * @param {{
     *   device?: object | null,
     *   shaderFallbackRouter?: object | null,
     *   pipelineCache?: object | null,
     *   format?: string,
     *   depthStencilFormat?: string | null,
     *   maxFallbackPipelines?: number,
     *   maxBindGroupCacheEntries?: number,
     *   enableShaderFallback?: boolean,
     *   enableBindGroupCache?: boolean,
     *   elideRedundantState?: boolean,
     *   elideRedundantPipelines?: boolean,
     *   elideRedundantBindGroups?: boolean,
     *   elideRedundantVertexBuffers?: boolean,
     *   elideRedundantIndexBuffers?: boolean,
     *   elideRedundantViewportScissor?: boolean,
     *   elideRedundantBlendStencil?: boolean,
     *   onWarning?: (message: string, meta?: object) => void,
     * }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('DrawExecutor options must be an object');
      }
      const opts = options || {};

      this._device = opts.device !== undefined ? opts.device : null;
      this._pipelineCache = opts.pipelineCache !== undefined ? opts.pipelineCache : null;
      this._format =
        typeof opts.format === 'string' && opts.format.length > 0
          ? opts.format
          : DEFAULT_COLOR_FORMAT;
      this._depthStencilFormat =
        typeof opts.depthStencilFormat === 'string' && opts.depthStencilFormat.length > 0
          ? opts.depthStencilFormat
          : null;

      const maxPipelines =
        opts.maxFallbackPipelines !== undefined
          ? opts.maxFallbackPipelines
          : DEFAULT_MAX_FALLBACK_PIPELINES;
      if (!Number.isInteger(maxPipelines) || maxPipelines < 1) {
        throw new RangeError(
          `maxFallbackPipelines must be a positive integer, got ${maxPipelines}`
        );
      }
      this._maxFallbackPipelines = maxPipelines;

      const maxBindGroupCache =
        opts.maxBindGroupCacheEntries !== undefined
          ? opts.maxBindGroupCacheEntries
          : DEFAULT_MAX_BIND_GROUP_CACHE_ENTRIES;
      if (!Number.isInteger(maxBindGroupCache) || maxBindGroupCache < 1) {
        throw new RangeError(
          `maxBindGroupCacheEntries must be a positive integer, got ${maxBindGroupCache}`
        );
      }
      this._maxBindGroupCacheEntries = maxBindGroupCache;
      this._enableBindGroupCache = opts.enableBindGroupCache !== false;
      this._enableShaderFallback = opts.enableShaderFallback !== false;

      const defaultElide = opts.elideRedundantState !== false;
      this._elideRedundantPipelines =
        opts.elideRedundantPipelines !== undefined
          ? Boolean(opts.elideRedundantPipelines)
          : defaultElide;
      this._elideRedundantBindGroups =
        opts.elideRedundantBindGroups !== undefined
          ? Boolean(opts.elideRedundantBindGroups)
          : defaultElide;
      this._elideRedundantVertexBuffers =
        opts.elideRedundantVertexBuffers !== undefined
          ? Boolean(opts.elideRedundantVertexBuffers)
          : defaultElide;
      this._elideRedundantIndexBuffers =
        opts.elideRedundantIndexBuffers !== undefined
          ? Boolean(opts.elideRedundantIndexBuffers)
          : defaultElide;
      this._elideRedundantViewportScissor =
        opts.elideRedundantViewportScissor !== undefined
          ? Boolean(opts.elideRedundantViewportScissor)
          : defaultElide;
      this._elideRedundantBlendStencil =
        opts.elideRedundantBlendStencil !== undefined
          ? Boolean(opts.elideRedundantBlendStencil)
          : defaultElide;
      this._reuseDrawResult = opts.reuseDrawResult === true;
      this._onWarning = typeof opts.onWarning === 'function' ? opts.onWarning : null;
      this._indirectClampExecutor =
        opts.indirectClampExecutor !== undefined ? opts.indirectClampExecutor : null;
      this._ownsIndirectClampExecutor = false;
      this._indirectFirstInstanceOverride =
        opts.indirectFirstInstance !== undefined ? Boolean(opts.indirectFirstInstance) : null;
      this._maxComputeWorkgroupsOverride =
        typeof opts.maxComputeWorkgroupsPerDimension === 'number' &&
        Number.isInteger(opts.maxComputeWorkgroupsPerDimension) &&
        opts.maxComputeWorkgroupsPerDimension >= 1
          ? opts.maxComputeWorkgroupsPerDimension
          : null;

      if (opts.shaderFallbackRouter !== undefined) {
        this._shaderFallbackRouter = opts.shaderFallbackRouter;
      } else if (this._enableShaderFallback) {
        const RouterCtor = resolveShaderFallbackRouterCtor();
        this._shaderFallbackRouter = RouterCtor ? new RouterCtor() : null;
      } else {
        this._shaderFallbackRouter = null;
      }

      // Active per-pass draw state (fixed-slot parallel arrays + bitmasks for O(1) dirty elision).
      this._activePipeline = null;
      this._activeComputePipeline = null;
      this._boundPass = null;
      this._boundPipeline = null;

      // Fast bind-group slots 0..3 (WebGPU guaranteed minimum) + lazy overflow Map for slots >= 4.
      this._stagedBgMask = 0;
      this._stagedBgRaw = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._stagedBgGroups = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._stagedBgMeta = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._stagedBgHasDyn = new Uint8Array(MAX_FAST_BG_SLOTS);
      this._stagedBgDynSource = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._stagedBgDynStart = new Int32Array(MAX_FAST_BG_SLOTS);
      this._stagedBgDynLength = new Int32Array(MAX_FAST_BG_SLOTS);
      this._stagedBgOverflow = null;

      this._boundBgMask = 0;
      this._dirtyBgMask = 0;
      this._boundBgGroups = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._boundBgHasDyn = new Uint8Array(MAX_FAST_BG_SLOTS);
      this._boundBgDynLen = new Int32Array(MAX_FAST_BG_SLOTS);
      this._boundBgDynOffsets = new Float64Array(MAX_FAST_BG_SLOTS * MAX_FAST_DYN_OFFSETS);
      this._boundBgDynOverflow = new Array(MAX_FAST_BG_SLOTS).fill(null);
      this._boundBgOverflow = null;

      // Fast vertex-buffer slots 0..7 (WebGPU guaranteed minimum) + lazy overflow Map for slots >= 8.
      this._stagedVbMask = 0;
      this._stagedVbBuffers = new Array(MAX_FAST_VB_SLOTS).fill(null);
      this._stagedVbOffsets = new Float64Array(MAX_FAST_VB_SLOTS);
      this._stagedVbSizes = new Array(MAX_FAST_VB_SLOTS).fill(null);
      this._stagedVbStrides = new Array(MAX_FAST_VB_SLOTS).fill(null);
      this._stagedVbOverflow = null;

      this._boundVbMask = 0;
      this._dirtyVbMask = 0;
      this._boundVbBuffers = new Array(MAX_FAST_VB_SLOTS).fill(null);
      this._boundVbOffsets = new Float64Array(MAX_FAST_VB_SLOTS);
      this._boundVbSizes = new Array(MAX_FAST_VB_SLOTS).fill(null);
      this._boundVbOverflow = null;

      // Preallocated staged & bound index-buffer / dynamic-state slots (zero per-call allocation).
      this._hasStagedIb = false;
      this._stagedIbBuffer = null;
      this._stagedIbFormat = 'uint16';
      this._stagedIbOffset = 0;
      this._stagedIbSize = null;
      this._hasBoundIb = false;
      this._dirtyIb = false;
      this._boundIbBuffer = null;
      this._boundIbFormat = null;
      this._boundIbOffset = 0;
      this._boundIbSize = null;

      this._hasStagedViewport = false;
      this._stagedViewport = { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 };
      this._hasBoundViewport = false;
      this._boundViewportData = { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 };
      this._tmpViewport = { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 };

      this._hasStagedScissor = false;
      this._stagedScissor = { x: 0, y: 0, width: 0, height: 0 };
      this._hasBoundScissor = false;
      this._boundScissorData = { x: 0, y: 0, width: 0, height: 0 };
      this._tmpScissor = { x: 0, y: 0, width: 0, height: 0 };

      this._hasStagedBlendConstant = false;
      this._stagedBlendConstant = { r: 0, g: 0, b: 0, a: 1 };
      this._hasBoundBlendConstant = false;
      this._boundBlendConstantData = { r: 0, g: 0, b: 0, a: 1 };
      this._tmpBlendConstant = { r: 0, g: 0, b: 0, a: 1 };

      this._stencilReference = null;
      this._boundStencilReference = null;

      // Fallback auxiliary resource maps.
      this._cbvs = new Map();
      this._textures = new Map();
      this._samplers = new Map();

      // Reusable draw/compute result & internal hot-path scratch structs (zero-alloc paths).
      this._reusableDrawResult = createDrawResultObject(false, false, 0, 1, 0);
      this._reusableIndirectResult = createDrawResultObject(false, true, 0, 0, 0);
      this._reusableComputeResult = {
        executed: false,
        skipped: false,
        reason: null,
        indirect: false,
        indirectOffset: 0,
        workgroupCountX: 0,
        workgroupCountY: 0,
        workgroupCountZ: 0,
        pipelineElided: false,
        bindGroupsElided: 0,
        bindGroupCacheHits: 0,
        bindGroupCacheMisses: 0,
      };
      this._tmpFallbackBgResult = {
        ok: false,
        substitutedCbv: false,
        substitutedTexture: false,
        substitutedSampler: false,
        cacheHit: false,
        bindGroupElided: false,
      };
      this._tmpFallbackVbResult = {
        ok: false,
        substitutedVertexBuffer: false,
        vertexBufferElided: false,
        effectiveVertexCount: 0,
        effectiveFirstVertex: 0,
      };
      this._reusableFrameSummary = {
        hasNonClearDraw: false,
        realPackDrawsThisFrame: 0,
        fallbackDrawsThisFrame: 0,
        drawsThisFrame: 0,
      };
      this._scratchClampOptions = {
        srcBuffer: null,
        srcOffset: 0,
        dstBuffer: null,
        dstOffset: 0,
        countBuffer: null,
        countOffset: 0,
        commandCount: 1,
        stride: undefined,
        maxVertexCount: undefined,
        maxIndexCount: undefined,
        maxInstanceCount: undefined,
        maxWorkgroupsX: undefined,
        maxWorkgroupsY: undefined,
        maxWorkgroupsZ: undefined,
        maxMeshVerticesPerGroup: undefined,
        maxMeshTotalVertices: undefined,
        indirectFirstInstance: undefined,
        maxIndirectBatchCount: undefined,
        clampBatch: true,
      };
      this._scratchClampResult = {
        ok: false,
        encoded: false,
        kind: 'draw',
        commandCount: 0,
        workgroupsX: 0,
        dstBuffer: null,
        dstOffset: 0,
        dstByteLength: 0,
        recordBytes: 0,
        stride: 0,
        zeroedFirstInstance: false,
        hasCountBuffer: false,
        cacheHit: false,
        reason: null,
      };
      this._reusableMeshResult = {
        ok: false,
        executed: false,
        skipped: false,
        elided: false,
        reason: null,
        indirectDispatch: false,
        dispatchClamped: false,
        drawClamped: false,
        indexed: false,
        drawsIssued: 0,
        drawIndirectBuffer: null,
        drawIndirectOffset: 0,
        zeroedFirstInstance: false,
      };
      this._reusableMultiPassResult = {
        ok: false,
        executed: 0,
        skipped: 0,
        elided: 0,
        clamped: 0,
        fallbackDirect: 0,
        firstInstanceClamped: 0,
        reason: null,
      };
      this._scratchMeshComputeArgs = {
        pipeline: null,
        bindGroups: null,
        workgroupCountX: 0,
        workgroupCountY: 1,
        workgroupCountZ: 1,
        indirect: false,
        indirectBuffer: null,
        indirectOffset: 0,
        reuseResult: true,
      };
      this._scratchMeshDrawArgs = {
        pipeline: null,
        indirect: true,
        indirectBuffer: null,
        indirectOffset: 0,
        reuseResult: true,
      };
      this._scratchReuseDrawOpts = { reuseResult: true };

      // Device-scoped resource caches & object identity maps (WeakMap avoids retention leaks).
      this._fallbackPipelineCache = new Map();
      this._fallbackShaderModuleCache = new Map();
      // W354: O(1) hash-bucketed LRU bind-group cache + WeakMap reverse resource index
      // (zero string allocation on 1..8 entry lookups, O(1) LRU promotion/eviction & invalidation).
      this._bgBuckets = new Map();
      this._bgHead = null;
      this._bgTail = null;
      this._bgCacheSize = 0;
      this._bgResourceIndex = new WeakMap();
      this._bgScratchBindings = new Int32Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchKinds = new Uint8Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchResA = new Array(MAX_FAST_BG_CACHE_ENTRIES).fill(null);
      this._bgScratchResB = new Array(MAX_FAST_BG_CACHE_ENTRIES).fill(null);
      this._bgScratchIdA = new Int32Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchIdB = new Int32Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchGenA = new Int32Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchGenB = new Int32Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchOff = new Float64Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._bgScratchSize = new Float64Array(MAX_FAST_BG_CACHE_ENTRIES);
      this._wgslHashCache = new Map();
      this._resourceIds = new WeakMap();
      this._nextResourceId = 0;
      this._textureViewCache = new WeakMap();
      this._bindGroupLayoutCache = new WeakMap();
      this._pipelineMetaCache = new WeakMap();

      this._fallbackCbvBuffer = null;
      this._fallbackTexture = null;
      this._fallbackTextureView = null;
      this._fallbackSampler = null;
      this._fallbackVertexBuffer24B = null;
      this._fallbackSplashVertexBuffer24B = null;

      this._destroyed = false;
      this._frameHasNonClearDraw = false;
      this._frameRealPackDraws = 0;
      this._frameFallbackDraws = 0;
      this._frameDraws = 0;
      this._totalStateCalls = 0;
      this._stats = createZeroedStats();
    }

    _warn(message, meta) {
      if (this._onWarning) {
        try {
          this._onWarning(message, meta);
        } catch (_err) {
          // Never allow diagnostic callbacks to break draw execution.
        }
      }
    }

    _getResourceId(resource) {
      if (!resource || (typeof resource !== 'object' && typeof resource !== 'function')) {
        return 0;
      }
      let id = this._resourceIds.get(resource);
      if (id === undefined) {
        this._nextResourceId += 1;
        id = this._nextResourceId;
        this._resourceIds.set(resource, id);
      }
      return id;
    }

    _hashWgsl(wgsl) {
      const str = typeof wgsl === 'string' ? wgsl : String(wgsl || '');
      const cached = this._wgslHashCache.get(str);
      if (cached !== undefined) {
        return cached;
      }
      const hash = fnv1a64Hex(str);
      if (this._wgslHashCache.size >= DEFAULT_MAX_WGSL_HASH_CACHE_ENTRIES) {
        const oldestKey = this._wgslHashCache.keys().next().value;
        this._wgslHashCache.delete(oldestKey);
      }
      this._wgslHashCache.set(str, hash);
      return hash;
    }

    /**
     * Look up or create a cached `GPUTextureView` for `(texture, viewDescriptor)` backed
     * by a `WeakMap<GPUTexture, { defaultView, byKey }>` so destroyed or replaced
     * textures are GC'd automatically with zero memory leaks.
     *
     * W347/W354: Respects promoted 1D->2D LUT view coercion (`dimension: '1d'` on a `'2d'` /
     * `promoted1DTo2D` texture -> `'2d'` with single layer/mip bounds) and the reverse 1D guard,
     * with zero descriptor object allocation on cache hits.
     *
     * @param {object} texture GPUTexture (or existing GPUTextureView)
     * @param {object | null | undefined} [viewDescriptor] GPUTextureViewDescriptor
     * @returns {object | null}
     */
    getOrCreateTextureView(texture, viewDescriptor) {
      if (!texture || typeof texture !== 'object') {
        return null;
      }
      if (texture.destroyed === true) {
        if (this._textureViewCache.has(texture)) {
          this._textureViewCache.delete(texture);
          this._stats.textureViewCacheEvictions += 1;
        }
        return null;
      }
      if (typeof texture.createView !== 'function') {
        if (texture.texture && texture.texture.destroyed === true) {
          return null;
        }
        return texture;
      }

      const hasDesc = Boolean(viewDescriptor && typeof viewDescriptor === 'object');
      let effFormat =
        hasDesc && viewDescriptor.format !== undefined && viewDescriptor.format !== null
          ? viewDescriptor.format
          : '';
      let effDim =
        hasDesc && viewDescriptor.dimension !== undefined && viewDescriptor.dimension !== null
          ? viewDescriptor.dimension
          : '';
      const rawAspect =
        hasDesc && viewDescriptor.aspect !== undefined && viewDescriptor.aspect !== null
          ? viewDescriptor.aspect
          : '';
      let effBaseMip =
        hasDesc && viewDescriptor.baseMipLevel !== undefined && viewDescriptor.baseMipLevel !== null
          ? viewDescriptor.baseMipLevel
          : '';
      let effMipCount =
        hasDesc &&
        viewDescriptor.mipLevelCount !== undefined &&
        viewDescriptor.mipLevelCount !== null
          ? viewDescriptor.mipLevelCount
          : '';
      let effBaseLayer =
        hasDesc &&
        viewDescriptor.baseArrayLayer !== undefined &&
        viewDescriptor.baseArrayLayer !== null
          ? viewDescriptor.baseArrayLayer
          : '';
      let effLayerCount =
        hasDesc &&
        viewDescriptor.arrayLayerCount !== undefined &&
        viewDescriptor.arrayLayerCount !== null
          ? viewDescriptor.arrayLayerCount
          : '';
      const rawUsage =
        hasDesc && viewDescriptor.usage !== undefined && viewDescriptor.usage !== null
          ? viewDescriptor.usage
          : '';

      let coerced = false;
      let formatCoerced = false;
      if (hasDesc && typeof effFormat === 'string' && effFormat.endsWith('-srgb')) {
        const texDesc = texture.desc && typeof texture.desc === 'object' ? texture.desc : null;
        const hasStorageBinding =
          (typeof texture.usage === 'number' && (texture.usage & 0x0008) !== 0) ||
          (texDesc !== null && typeof texDesc.usage === 'number' && (texDesc.usage & 0x0008) !== 0);
        let missingSrgbViewFormat = false;
        if (!hasStorageBinding) {
          const baseTexFmt =
            (typeof texture.format === 'string' && texture.format) ||
            (texDesc && typeof texDesc.format === 'string' && texDesc.format) ||
            null;
          if (baseTexFmt && !baseTexFmt.endsWith('-srgb')) {
            if (Array.isArray(texture.viewFormats)) {
              missingSrgbViewFormat = !texture.viewFormats.includes(effFormat);
            } else if (texDesc && Array.isArray(texDesc.viewFormats)) {
              missingSrgbViewFormat = !texDesc.viewFormats.includes(effFormat);
            } else if (texDesc && typeof texDesc.format === 'string' && !Array.isArray(texDesc.viewFormats)) {
              missingSrgbViewFormat = true;
            }
          }
        }
        if (hasStorageBinding || missingSrgbViewFormat) {
          effFormat = effFormat.slice(0, -5);
          formatCoerced = true;
          coerced = true;
        }
      }
      if (hasDesc && typeof effDim === 'string' && effDim.length > 0 && effDim !== 'undefined') {
        const actualTexDim =
          typeof texture.dimension === 'string'
            ? texture.dimension
            : texture.desc && typeof texture.desc.dimension === 'string'
              ? texture.desc.dimension
              : null;
        const isPromoted1DTo2D = Boolean(
          texture.promoted1DTo2D === true ||
            (texture.desc && texture.desc.promoted1DTo2D === true)
        );
        if (effDim === '1d' && (actualTexDim === '2d' || isPromoted1DTo2D)) {
          // W347: coerce '1d' view descriptor to '2d' with valid single-layer/mip bounds on promoted 1D->2D LUTs.
          const texMipCount =
            typeof texture.mipLevelCount === 'number' && texture.mipLevelCount >= 1
              ? texture.mipLevelCount
              : texture.desc &&
                  typeof texture.desc.mipLevelCount === 'number' &&
                  texture.desc.mipLevelCount >= 1
                ? texture.desc.mipLevelCount
                : isPromoted1DTo2D
                  ? 1
                  : null;
          effDim = '2d';
          effBaseLayer = 0;
          effLayerCount = 1;
          if (isPromoted1DTo2D || texMipCount === 1) {
            effBaseMip = 0;
            effMipCount = 1;
          }
          coerced = true;
        } else if (actualTexDim === '1d' && effDim !== '1d') {
          // W347: reverse guard — real WebGPU '1d' textures only allow '1d' views (1 mip, 1 layer).
          effDim = '1d';
          if (effBaseLayer !== '' && effBaseLayer !== 0) effBaseLayer = 0;
          if (effLayerCount !== '' && effLayerCount !== 1) effLayerCount = 1;
          if (effBaseMip !== '' && effBaseMip !== 0) effBaseMip = 0;
          if (effMipCount !== '' && effMipCount !== 1) effMipCount = 1;
          coerced = true;
        }
      }

      const isDefaultKey =
        effFormat === '' &&
        effDim === '' &&
        rawAspect === '' &&
        effBaseMip === '' &&
        effMipCount === '' &&
        effBaseLayer === '' &&
        effLayerCount === '' &&
        rawUsage === '';

      let record = this._textureViewCache.get(texture);

      if (isDefaultKey) {
        if (record && record.defaultView) {
          if (record.defaultView.destroyed !== true) {
            this._stats.textureViewCacheHits += 1;
            return record.defaultView;
          }
          record.defaultView = null;
          this._stats.textureViewCacheEvictions += 1;
        }
        this._stats.textureViewCacheMisses += 1;
        try {
          const view = texture.createView();
          if (view && typeof view === 'object') {
            if (!record) {
              record = { defaultView: view, byKey: null };
              this._textureViewCache.set(texture, record);
            } else {
              record.defaultView = view;
            }
          }
          return view;
        } catch (_err) {
          return null;
        }
      }

      const key = `${effFormat}|${effDim}|${rawAspect}|${effBaseMip}|${effMipCount}|${effBaseLayer}|${effLayerCount}|${rawUsage}`;
      if (record && record.byKey !== null) {
        const cached = record.byKey.get(key);
        if (cached) {
          if (cached.destroyed !== true) {
            this._stats.textureViewCacheHits += 1;
            return cached;
          }
          record.byKey.delete(key);
          this._stats.textureViewCacheEvictions += 1;
        }
      }

      this._stats.textureViewCacheMisses += 1;
      let createDesc = viewDescriptor;
      if (coerced) {
        createDesc = { ...viewDescriptor };
        if (formatCoerced) createDesc.format = effFormat;
        if (effDim !== '') createDesc.dimension = effDim;
        if (effBaseLayer !== '') createDesc.baseArrayLayer = effBaseLayer;
        if (effLayerCount !== '') createDesc.arrayLayerCount = effLayerCount;
        if (effBaseMip !== '') createDesc.baseMipLevel = effBaseMip;
        if (effMipCount !== '') createDesc.mipLevelCount = effMipCount;
      }

      try {
        const view = texture.createView(createDesc);
        if (view && typeof view === 'object') {
          if (!record) {
            const byKey = new Map();
            byKey.set(key, view);
            this._textureViewCache.set(texture, { defaultView: null, byKey });
          } else {
            if (record.byKey === null) {
              record.byKey = new Map();
            } else if (record.byKey.size >= MAX_VIEWS_PER_TEXTURE) {
              const oldestKey = record.byKey.keys().next().value;
              record.byKey.delete(oldestKey);
              this._stats.textureViewCacheEvictions += 1;
            }
            record.byKey.set(key, view);
          }
        }
        return view;
      } catch (_err) {
        return null;
      }
    }

    _resolveTextureViewCached(resource, viewDescriptor) {
      if (!isValidTextureOrView(resource)) {
        return null;
      }
      if (resource.textureView && isValidTextureOrView(resource.textureView)) {
        if (resource.texture && resource.texture.destroyed === true) {
          return null;
        }
        return resource.textureView;
      }
      if (resource.texture) {
        const tex = resource.texture;
        if (!isValidTextureOrView(tex)) {
          if (this._textureViewCache.has(tex)) {
            this._textureViewCache.delete(tex);
            this._stats.textureViewCacheEvictions += 1;
          }
          return null;
        }
        const desc =
          viewDescriptor !== undefined
            ? viewDescriptor
            : resource.viewDescriptor !== undefined
              ? resource.viewDescriptor
              : resource.viewDesc;
        return this.getOrCreateTextureView(tex, desc);
      }
      if (typeof resource.createView === 'function') {
        return this.getOrCreateTextureView(resource, viewDescriptor);
      }
      return resource;
    }

    _getPipelineBindGroupLayout(pipeline, groupIndex) {
      let groupMap = this._bindGroupLayoutCache.get(pipeline);
      if (groupMap && groupMap.has(groupIndex)) {
        return groupMap.get(groupIndex);
      }
      const layout = pipeline.getBindGroupLayout(groupIndex);
      if (layout) {
        if (!groupMap) {
          groupMap = new Map();
          this._bindGroupLayoutCache.set(pipeline, groupMap);
        }
        groupMap.set(groupIndex, layout);
      }
      return layout;
    }

    _getPipelineMetaFlags(pipe) {
      if (!pipe || typeof pipe !== 'object') {
        return 0;
      }
      const cached = this._pipelineMetaCache.get(pipe);
      if (cached !== undefined) {
        return cached;
      }
      let flags = 0;
      if (hasAlphaBlendDescriptor(pipe)) {
        flags |= PIPE_FLAG_ALPHA_BLEND;
      }
      const label = typeof pipe.label === 'string' ? pipe.label.toLowerCase() : '';
      const cat =
        typeof pipe.drawCategory === 'string'
          ? pipe.drawCategory
          : typeof pipe.category === 'string'
            ? pipe.category
            : '';
      if (
        pipe.isFontGlyph === true ||
        pipe.isGlyph === true ||
        cat === 'font-glyph' ||
        label.includes('glyph') ||
        label.includes('font')
      ) {
        flags |= PIPE_FLAG_FONT_GLYPH;
      }
      if (
        pipe.isVideo === true ||
        pipe.isIntroVideo === true ||
        cat === 'video' ||
        label.includes('video') ||
        label.includes('bink') ||
        label.includes('nv12')
      ) {
        flags |= PIPE_FLAG_VIDEO;
      }
      if (
        pipe.isUiQuad === true ||
        pipe.isUi === true ||
        cat === 'ui-quad' ||
        label.includes('ui_quad') ||
        label.includes('ui-quad') ||
        label.includes('menu') ||
        label.includes('hud')
      ) {
        flags |= PIPE_FLAG_UI_QUAD;
      }
      if (pipe.isRectList === true || isRectListTopology(pipe.topology)) {
        flags |= PIPE_FLAG_RECTLIST;
      }
      if (pipe.vertexStride === TEXTURED_QUAD_VERTEX_STRIDE) {
        flags |= PIPE_FLAG_STRIDE_24;
      }
      if (pipe.isClear === true || pipe.isClearDraw === true || cat === 'clear') {
        flags |= PIPE_FLAG_CLEAR;
      }
      this._pipelineMetaCache.set(pipe, flags);
      return flags;
    }

    _recordExecutedDrawCategory(args, pipe, count, effectiveProfile) {
      const pipeFlags = pipe && typeof pipe === 'object' ? this._getPipelineMetaFlags(pipe) : 0;
      const cat =
        args && typeof args.drawCategory === 'string'
          ? args.drawCategory
          : args && typeof args.category === 'string'
            ? args.category
            : null;
      const isAlphaBlended =
        (args && (args.alphaBlended === true || args.blendEnabled === true)) ||
        (pipeFlags & PIPE_FLAG_ALPHA_BLEND) !== 0;
      const isFontGlyph =
        (args && (args.isFontGlyph === true || args.isGlyph === true)) ||
        cat === 'font-glyph' ||
        (pipeFlags & PIPE_FLAG_FONT_GLYPH) !== 0;
      const isVideo =
        (args && (args.isVideo === true || args.isIntroVideo === true)) ||
        cat === 'video' ||
        (pipeFlags & PIPE_FLAG_VIDEO) !== 0;
      const isQuadCount =
        count === 4 ||
        count === 6 ||
        (count === 3 &&
          ((pipeFlags & PIPE_FLAG_RECTLIST) !== 0 ||
            (args && (args.isRectList === true || isRectListTopology(args.topology)))));
      const isUiQuad =
        (args && (args.isUiQuad === true || args.isUi === true)) ||
        cat === 'ui-quad' ||
        (pipeFlags & PIPE_FLAG_UI_QUAD) !== 0 ||
        effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B ||
        effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B ||
        (isAlphaBlended && !isFontGlyph && !isVideo && isQuadCount);

      if (isAlphaBlended) {
        this._stats.alphaBlendedDraws += 1;
      }
      if (isFontGlyph) {
        this._stats.fontGlyphDraws += 1;
      }
      if (isUiQuad) {
        this._stats.uiQuadDraws += 1;
      }
      if (isVideo) {
        this._stats.videoDraws += 1;
      }
    }

    _recordExecutedDrawVisibility(args, pipe, isFallback) {
      this._stats.totalDrawsExecuted += 1;
      this._frameDraws += 1;
      if (isFallback) {
        this._frameFallbackDraws += 1;
      } else {
        this._frameRealPackDraws += 1;
      }
      const pipeFlags = pipe && typeof pipe === 'object' ? this._getPipelineMetaFlags(pipe) : 0;
      const isClearDraw =
        (pipeFlags & PIPE_FLAG_CLEAR) !== 0 ||
        (args !== null &&
          args !== undefined &&
          (args.isClear === true ||
            args.isClearDraw === true ||
            args.drawCategory === 'clear' ||
            args.category === 'clear'));
      if (!isClearDraw) {
        this._frameHasNonClearDraw = true;
      }
    }

    _computeStateElisionHitRate() {
      const total = this._totalStateCalls;
      if (!(total > 0)) {
        return 0;
      }
      const raw = this._stats.totalElidedStateCalls / total;
      if (!Number.isFinite(raw) || raw <= 0) {
        return 0;
      }
      return raw >= 1 ? 1 : raw;
    }

    _recordSkip(reason, invalidRange) {
      this._stats.skippedDraws += 1;
      const key = typeof reason === 'string' && reason.length > 0 ? reason : 'unknown';
      this._stats.skippedByReason[key] = (this._stats.skippedByReason[key] || 0) + 1;
      const isInvalidRange =
        invalidRange === true || (invalidRange && invalidRange.invalidRange === true);

      if (key === 'executor-destroyed') {
        this._stats.skippedDestroyed += 1;
      } else if (key === 'invalid-render-pass') {
        this._stats.skippedNoPass += 1;
      } else if (key === 'zero-draw-count') {
        this._stats.zeroCountDrawsSkipped += 1;
        this._stats.skippedInvalidRange += 1;
      } else if (
        key === 'no-pipeline' ||
        key === 'shader-resolve-failed' ||
        key === 'pipeline-creation-failed'
      ) {
        this._stats.skippedNoPipeline += 1;
      } else if (key === 'invalid-index-buffer') {
        this._stats.skippedInvalidIndexBuffer += 1;
        if (isInvalidRange) {
          this._stats.skippedInvalidRange += 1;
        }
      } else if (key === 'invalid-indirect-buffer') {
        this._stats.skippedInvalidIndirectBuffer += 1;
        if (isInvalidRange) {
          this._stats.skippedInvalidRange += 1;
        }
      } else if (
        key === 'fallback-cbv-bind-failed' ||
        key === 'fallback-texture-bind-failed' ||
        key === 'fallback-splash-bind-failed' ||
        key === 'fallback-vb-bind-failed'
      ) {
        this._stats.skippedBindFailed += 1;
      } else if (key === 'pass-error') {
        this._stats.skippedPassError += 1;
      }
    }

    _clearBoundBindGroups() {
      let bgMask = this._boundBgMask;
      while (bgMask !== 0) {
        const bit = bgMask & -bgMask;
        const slot = 31 - Math.clz32(bit);
        bgMask ^= bit;
        this._boundBgGroups[slot] = null;
        this._boundBgHasDyn[slot] = 0;
        this._boundBgDynLen[slot] = 0;
      }
      this._boundBgMask = 0;
      this._dirtyBgMask = this._stagedBgMask;
      if (this._boundBgOverflow !== null && this._boundBgOverflow.size > 0) {
        this._boundBgOverflow.clear();
      }
    }

    _clearBoundVertexBuffers() {
      let vbMask = this._boundVbMask;
      while (vbMask !== 0) {
        const bit = vbMask & -vbMask;
        const slot = 31 - Math.clz32(bit);
        vbMask ^= bit;
        this._boundVbBuffers[slot] = null;
        this._boundVbOffsets[slot] = 0;
        this._boundVbSizes[slot] = null;
      }
      this._boundVbMask = 0;
      this._dirtyVbMask = this._stagedVbMask;
      if (this._boundVbOverflow !== null && this._boundVbOverflow.size > 0) {
        this._boundVbOverflow.clear();
      }
    }

    _clearBoundPassState() {
      this._boundPass = null;
      this._boundPipeline = null;
      this._clearBoundBindGroups();
      this._clearBoundVertexBuffers();
      this._hasBoundIb = false;
      this._boundIbBuffer = null;
      this._boundIbFormat = null;
      this._boundIbOffset = 0;
      this._boundIbSize = null;
      this._dirtyIb = this._hasStagedIb;
      this._hasBoundViewport = false;
      this._hasBoundScissor = false;
      this._hasBoundBlendConstant = false;
      this._boundStencilReference = null;
    }

    _syncPassTracking(pass) {
      if (this._boundPass !== pass) {
        this._clearBoundPassState();
        this._boundPass = pass;
      }
    }

    _clearBgCacheInternal() {
      this._bgBuckets.clear();
      this._bgHead = null;
      this._bgTail = null;
      this._bgCacheSize = 0;
      this._bgResourceIndex = new WeakMap();
    }

    /**
     * Clear the device-scoped bind-group reuse cache and derived view/layout caches.
     */
    clearCache() {
      this._clearBgCacheInternal();
      this._textureViewCache = new WeakMap();
      this._bindGroupLayoutCache = new WeakMap();
      this._clearBoundBindGroups();
    }

    /**
     * Alias for `clearCache()` preserved for backwards compatibility.
     */
    clearBindGroupCache() {
      this.clearCache();
    }

    _bgLinkToHead(node) {
      node.lruPrev = null;
      node.lruNext = this._bgHead;
      if (this._bgHead !== null) {
        this._bgHead.lruPrev = node;
      } else {
        this._bgTail = node;
      }
      this._bgHead = node;
    }

    _bgPromoteToHead(node) {
      if (this._bgHead === node) {
        return;
      }
      const prev = node.lruPrev;
      const next = node.lruNext;
      if (prev !== null) prev.lruNext = next;
      if (next !== null) next.lruPrev = prev;
      if (this._bgTail === node) this._bgTail = prev;
      node.lruPrev = null;
      node.lruNext = this._bgHead;
      if (this._bgHead !== null) {
        this._bgHead.lruPrev = node;
      } else {
        this._bgTail = node;
      }
      this._bgHead = node;
    }

    _bgUnlinkNode(node) {
      // 1. Unlink from hash bucket chain.
      const h = node.hash;
      const bPrev = node.bucketPrev;
      const bNext = node.bucketNext;
      if (bPrev !== null) {
        bPrev.bucketNext = bNext;
      } else if (bNext !== null) {
        this._bgBuckets.set(h, bNext);
      } else {
        this._bgBuckets.delete(h);
      }
      if (bNext !== null) {
        bNext.bucketPrev = bPrev;
      }
      node.bucketPrev = null;
      node.bucketNext = null;

      // 2. Unlink from doubly-linked LRU list.
      const lPrev = node.lruPrev;
      const lNext = node.lruNext;
      if (lPrev !== null) lPrev.lruNext = lNext;
      else if (this._bgHead === node) this._bgHead = lNext;
      if (lNext !== null) lNext.lruPrev = lPrev;
      else if (this._bgTail === node) this._bgTail = lPrev;
      node.lruPrev = null;
      node.lruNext = null;

      // 3. Remove from reverse resource index.
      const tracked = node.resources;
      if (tracked !== null) {
        for (let i = 0; i < tracked.length; i += 1) {
          const r = tracked[i];
          if (r && (typeof r === 'object' || typeof r === 'function')) {
            const set = this._bgResourceIndex.get(r);
            if (set !== undefined) {
              set.delete(node);
              if (set.size === 0) {
                this._bgResourceIndex.delete(r);
              }
            }
          }
        }
      }
      this._bgCacheSize -= 1;
    }

    _bgRegisterResourceIndex(node, tracked) {
      if (!tracked) return;
      for (let i = 0; i < tracked.length; i += 1) {
        const r = tracked[i];
        if (!r || (typeof r !== 'object' && typeof r !== 'function')) continue;
        let set = this._bgResourceIndex.get(r);
        if (set === undefined) {
          set = new Set();
          this._bgResourceIndex.set(r, set);
        }
        set.add(node);
      }
    }

    _bgNodeMatches(
      node,
      ownerKind,
      owner,
      groupIndex,
      count,
      bindings,
      kinds,
      resA,
      resB,
      genA,
      genB,
      offs,
      sizes
    ) {
      if (
        node.ownerKind !== ownerKind ||
        node.owner !== owner ||
        node.groupIndex !== groupIndex ||
        node.entryCount !== count
      ) {
        return false;
      }
      const nb = node.bindings;
      const nk = node.kinds;
      const na = node.resA;
      const nbRes = node.resB;
      const nga = node.genA;
      const ngb = node.genB;
      const no = node.offs;
      const ns = node.sizes;
      for (let i = 0; i < count; i += 1) {
        if (
          nb[i] !== bindings[i] ||
          nk[i] !== kinds[i] ||
          na[i] !== resA[i] ||
          nbRes[i] !== resB[i] ||
          nga[i] !== genA[i] ||
          ngb[i] !== genB[i] ||
          no[i] !== offs[i] ||
          ns[i] !== sizes[i]
        ) {
          return false;
        }
      }
      return true;
    }

    _bgLookupNode(
      hash,
      ownerKind,
      owner,
      groupIndex,
      count,
      bindings,
      kinds,
      resA,
      resB,
      genA,
      genB,
      offs,
      sizes
    ) {
      if (!this._enableBindGroupCache || this._bgCacheSize === 0) {
        return null;
      }
      let node = this._bgBuckets.get(hash);
      while (node !== undefined && node !== null) {
        const next = node.bucketNext;
        if (
          this._bgNodeMatches(
            node,
            ownerKind,
            owner,
            groupIndex,
            count,
            bindings,
            kinds,
            resA,
            resB,
            genA,
            genB,
            offs,
            sizes
          )
        ) {
          if (isCachedBindGroupValid(node)) {
            this._bgPromoteToHead(node);
            this._stats.bindGroupCacheHits += 1;
            return node.bindGroup;
          }
          this._bgUnlinkNode(node);
          this._stats.bindGroupCacheEvictions += 1;
          return null;
        }
        node = next;
      }
      return null;
    }

    _bgStoreNode(
      hash,
      ownerKind,
      owner,
      groupIndex,
      count,
      bindings,
      kinds,
      resA,
      resB,
      genA,
      genB,
      offs,
      sizes,
      bindGroup,
      trackedResources
    ) {
      if (!this._enableBindGroupCache || !bindGroup) {
        return;
      }
      while (this._bgCacheSize >= this._maxBindGroupCacheEntries && this._bgTail !== null) {
        this._bgUnlinkNode(this._bgTail);
        this._stats.bindGroupCacheEvictions += 1;
      }
      const cBindings = new Int32Array(count);
      const cKinds = new Uint8Array(count);
      const cResA = new Array(count);
      const cResB = new Array(count);
      const cGenA = new Int32Array(count);
      const cGenB = new Int32Array(count);
      const cOffs = new Float64Array(count);
      const cSizes = new Float64Array(count);
      for (let i = 0; i < count; i += 1) {
        cBindings[i] = bindings[i];
        cKinds[i] = kinds[i];
        cResA[i] = resA[i];
        cResB[i] = resB[i];
        cGenA[i] = genA[i];
        cGenB[i] = genB[i];
        cOffs[i] = offs[i];
        cSizes[i] = sizes[i];
      }
      let generations = null;
      if (trackedResources && trackedResources.length > 0) {
        let hasGen = false;
        for (let i = 0; i < trackedResources.length; i += 1) {
          const r = trackedResources[i];
          if (r && typeof r.generation === 'number') {
            hasGen = true;
            break;
          }
        }
        if (hasGen) {
          generations = new Array(trackedResources.length);
          for (let i = 0; i < trackedResources.length; i += 1) {
            generations[i] = getResourceGeneration(trackedResources[i]);
          }
        }
      }
      const existingHead = this._bgBuckets.get(hash) || null;
      const node = {
        hash,
        ownerKind,
        owner,
        groupIndex,
        entryCount: count,
        bindings: cBindings,
        kinds: cKinds,
        resA: cResA,
        resB: cResB,
        genA: cGenA,
        genB: cGenB,
        offs: cOffs,
        sizes: cSizes,
        bindGroup,
        resources: trackedResources || null,
        generations,
        bucketPrev: null,
        bucketNext: existingHead,
        lruPrev: null,
        lruNext: null,
      };
      if (existingHead !== null) {
        existingHead.bucketPrev = node;
      }
      this._bgBuckets.set(hash, node);
      this._bgLinkToHead(node);
      this._bgRegisterResourceIndex(node, trackedResources);
      this._bgCacheSize += 1;
    }

    /**
     * Invalidate all cached `GPUTextureView` and `GPUBindGroup` entries referencing
     * `resource` in O(1) via reverse WeakMap index (call when a texture, buffer, or
     * sampler is replaced or destroyed).
     *
     * @param {object | null | undefined} resource
     * @returns {number} Number of evicted cache entries
     */
    invalidateResource(resource) {
      if (!resource || typeof resource !== 'object') {
        return 0;
      }
      let evicted = 0;
      if (this._textureViewCache.has(resource)) {
        this._textureViewCache.delete(resource);
        this._stats.textureViewCacheEvictions += 1;
        evicted += 1;
      }
      const nodes = this._bgResourceIndex.get(resource);
      if (nodes !== undefined && nodes.size > 0) {
        const toEvict = Array.from(nodes);
        for (let i = 0; i < toEvict.length; i += 1) {
          this._bgUnlinkNode(toEvict[i]);
          this._stats.bindGroupCacheEvictions += 1;
          evicted += 1;
        }
        this._clearBoundBindGroups();
      }
      return evicted;
    }

    invalidateTexture(texture) {
      return this.invalidateResource(texture);
    }

    invalidateBuffer(buffer) {
      return this.invalidateResource(buffer);
    }

    _getOrCreateBindGroupStructured(ownerKind, owner, groupIndex, entries, label) {
      const count = entries.length;
      const useScratch = count <= MAX_FAST_BG_CACHE_ENTRIES;
      const bindings = useScratch ? this._bgScratchBindings : new Int32Array(count);
      const kinds = useScratch ? this._bgScratchKinds : new Uint8Array(count);
      const resA = useScratch ? this._bgScratchResA : new Array(count);
      const resB = useScratch ? this._bgScratchResB : new Array(count);
      const idA = useScratch ? this._bgScratchIdA : new Int32Array(count);
      const idB = useScratch ? this._bgScratchIdB : new Int32Array(count);
      const genA = useScratch ? this._bgScratchGenA : new Int32Array(count);
      const genB = useScratch ? this._bgScratchGenB : new Int32Array(count);
      const offs = useScratch ? this._bgScratchOff : new Float64Array(count);
      const sizes = useScratch ? this._bgScratchSize : new Float64Array(count);

      let needsSort = false;
      let anyTextureResolved = false;

      // Pass 1: validate entries, resolve GPUTexture -> cached GPUTextureView, populate flat
      // scratch slots with ZERO string, array, or object allocations on the 1..8 entry hit path.
      for (let i = 0; i < count; i += 1) {
        const entry = entries[i];
        if (!entry || typeof entry !== 'object') {
          if (useScratch) {
            for (let k = 0; k < i; k += 1) {
              resA[k] = null;
              resB[k] = null;
            }
          }
          this._stats.bindGroupErrors += 1;
          return null;
        }
        const bNum = entry.binding | 0;
        if (i > 0 && bNum < bindings[i - 1]) {
          needsSort = true;
        }
        bindings[i] = bNum;
        const res = entry.resource;
        if (res && typeof res === 'object' && res.buffer !== undefined) {
          const buf = res.buffer;
          if (!buf || buf.destroyed === true) {
            if (useScratch) {
              for (let k = 0; k < i; k += 1) {
                resA[k] = null;
                resB[k] = null;
              }
            }
            this._stats.bindGroupErrors += 1;
            return null;
          }
          kinds[i] = BG_ENTRY_KIND_BUFFER;
          resA[i] = buf;
          resB[i] = null;
          idA[i] = this._getResourceId(buf);
          idB[i] = 0;
          genA[i] = getResourceGeneration(buf);
          genB[i] = 0;
          offs[i] = res.offset !== undefined ? res.offset : 0;
          sizes[i] = res.size !== undefined && res.size !== null ? res.size : -1;
        } else if (
          res &&
          typeof res === 'object' &&
          (typeof res.createView === 'function' || res.texture !== undefined)
        ) {
          const srcTex = res.texture !== undefined ? res.texture : res;
          if (!srcTex || srcTex.destroyed === true) {
            if (srcTex && this._textureViewCache.has(srcTex)) {
              this._textureViewCache.delete(srcTex);
              this._stats.textureViewCacheEvictions += 1;
            }
            if (useScratch) {
              for (let k = 0; k < i; k += 1) {
                resA[k] = null;
                resB[k] = null;
              }
            }
            this._stats.bindGroupErrors += 1;
            return null;
          }
          const viewDesc =
            res.viewDescriptor !== undefined
              ? res.viewDescriptor
              : res.viewDesc !== undefined
                ? res.viewDesc
                : undefined;
          const view = this._resolveTextureViewCached(res, viewDesc);
          if (!view || view.destroyed === true) {
            if (useScratch) {
              for (let k = 0; k < i; k += 1) {
                resA[k] = null;
                resB[k] = null;
              }
            }
            this._stats.bindGroupErrors += 1;
            return null;
          }
          kinds[i] = BG_ENTRY_KIND_TEXTURE;
          resA[i] = srcTex;
          resB[i] = view;
          idA[i] = this._getResourceId(srcTex);
          idB[i] = this._getResourceId(view);
          genA[i] = getResourceGeneration(srcTex);
          genB[i] = getResourceGeneration(view);
          offs[i] = 0;
          sizes[i] = -1;
          anyTextureResolved = true;
        } else {
          if (!res || res.destroyed === true || (res.texture && res.texture.destroyed === true)) {
            if (useScratch) {
              for (let k = 0; k < i; k += 1) {
                resA[k] = null;
                resB[k] = null;
              }
            }
            this._stats.bindGroupErrors += 1;
            return null;
          }
          kinds[i] = BG_ENTRY_KIND_RAW;
          resA[i] = res;
          resB[i] = null;
          idA[i] = this._getResourceId(res);
          idB[i] = 0;
          genA[i] = getResourceGeneration(res);
          genB[i] = 0;
          offs[i] = 0;
          sizes[i] = -1;
        }
      }

      // In-place insertion sort by binding number ONLY when out-of-order (zero allocations).
      if (needsSort) {
        for (let i = 1; i < count; i += 1) {
          const kb = bindings[i];
          const kk = kinds[i];
          const kra = resA[i];
          const krb = resB[i];
          const kia = idA[i];
          const kib = idB[i];
          const kga = genA[i];
          const kgb = genB[i];
          const ko = offs[i];
          const ks = sizes[i];
          let j = i - 1;
          while (j >= 0 && bindings[j] > kb) {
            bindings[j + 1] = bindings[j];
            kinds[j + 1] = kinds[j];
            resA[j + 1] = resA[j];
            resB[j + 1] = resB[j];
            idA[j + 1] = idA[j];
            idB[j + 1] = idB[j];
            genA[j + 1] = genA[j];
            genB[j + 1] = genB[j];
            offs[j + 1] = offs[j];
            sizes[j + 1] = sizes[j];
            j -= 1;
          }
          bindings[j + 1] = kb;
          kinds[j + 1] = kk;
          resA[j + 1] = kra;
          resB[j + 1] = krb;
          idA[j + 1] = kia;
          idB[j + 1] = kib;
          genA[j + 1] = kga;
          genB[j + 1] = kgb;
          offs[j + 1] = ko;
          sizes[j + 1] = ks;
        }
      }

      let h = -2128831035 | 0; // FNV-1a 32-bit offset basis
      h = hashMixInt32(h, ownerKind);
      h = hashMixInt32(h, this._getResourceId(owner));
      h = hashMixInt32(h, groupIndex);
      h = hashMixInt32(h, count);
      for (let i = 0; i < count; i += 1) {
        h = hashMixInt32(h, bindings[i]);
        h = hashMixInt32(h, kinds[i]);
        h = hashMixInt32(h, idA[i]);
        h = hashMixInt32(h, genA[i]);
        if (kinds[i] === BG_ENTRY_KIND_BUFFER) {
          h = hashMixNum(h, offs[i]);
          h = hashMixNum(h, sizes[i]);
        } else if (kinds[i] === BG_ENTRY_KIND_TEXTURE) {
          h = hashMixInt32(h, idB[i]);
          h = hashMixInt32(h, genB[i]);
        }
      }

      const cachedBg = this._bgLookupNode(
        h,
        ownerKind,
        owner,
        groupIndex,
        count,
        bindings,
        kinds,
        resA,
        resB,
        genA,
        genB,
        offs,
        sizes
      );
      if (cachedBg !== null) {
        if (useScratch) {
          for (let k = 0; k < count; k += 1) {
            resA[k] = null;
            resB[k] = null;
          }
        }
        return cachedBg;
      }

      // Cache miss: build resolved entries (only if any GPUTexture was passed) using the already-resolved
      // views in resB (zero duplicate _resolveTextureViewCached calls) and collect trackedResources.
      let createEntries = entries;
      if (anyTextureResolved) {
        createEntries = new Array(count);
        for (let i = 0; i < count; i += 1) {
          const entry = entries[i];
          const res = entry.resource;
          if (
            res &&
            typeof res === 'object' &&
            (typeof res.createView === 'function' || res.texture !== undefined)
          ) {
            let view = null;
            if (!needsSort) {
              view = resB[i];
            } else {
              const targetBinding = entry.binding | 0;
              for (let j = 0; j < count; j += 1) {
                if (bindings[j] === targetBinding && kinds[j] === BG_ENTRY_KIND_TEXTURE) {
                  view = resB[j];
                  break;
                }
              }
            }
            createEntries[i] = { binding: entry.binding, resource: view };
          } else {
            createEntries[i] = entry;
          }
        }
      }

      const trackedResources = [];
      for (let i = 0; i < count; i += 1) {
        const k = kinds[i];
        const ra = resA[i];
        if (k === BG_ENTRY_KIND_BUFFER) {
          trackedResources.push(ra);
        } else if (k === BG_ENTRY_KIND_TEXTURE) {
          trackedResources.push(ra);
          const rb = resB[i];
          if (rb && rb !== ra) {
            trackedResources.push(rb);
          }
        } else if (ra) {
          trackedResources.push(ra);
          if (ra.texture && ra.texture !== ra) {
            trackedResources.push(ra.texture);
          }
        }
      }

      try {
        this._stats.bindGroupCacheMisses += 1;
        const layout =
          ownerKind === BG_OWNER_LAYOUT
            ? owner
            : this._getPipelineBindGroupLayout(owner, groupIndex);
        const effectiveLabel =
          typeof label === 'string' && label.length > 0
            ? label
            : ownerKind === BG_OWNER_LAYOUT
              ? 'draw_executor:cached_bg'
              : `draw_executor:pipe_bg_${groupIndex}`;
        const bindGroup = this._device.createBindGroup({
          label: effectiveLabel,
          layout,
          entries: createEntries,
        });
        this._bgStoreNode(
          h,
          ownerKind,
          owner,
          groupIndex,
          count,
          bindings,
          kinds,
          resA,
          resB,
          genA,
          genB,
          offs,
          sizes,
          bindGroup,
          trackedResources
        );
        if (useScratch) {
          for (let k = 0; k < count; k += 1) {
            resA[k] = null;
            resB[k] = null;
          }
        }
        return bindGroup;
      } catch (err) {
        if (useScratch) {
          for (let k = 0; k < count; k += 1) {
            resA[k] = null;
            resB[k] = null;
          }
        }
        this._stats.bindGroupErrors += 1;
        this._warn('getOrCreateBindGroup failed', {
          label,
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    /**
     * Look up or create a cached GPUBindGroup keyed by `(layout, groupIndex?, entryResources)`.
     * Automatically resolves `GPUTexture` / `{ texture, viewDescriptor }` entries through the
     * per-`(texture, viewDescriptorKey)` view cache and invalidates cached bind groups when
     * backing resources are destroyed or change generation.
     *
     * @param {object} layout GPUBindGroupLayout
     * @param {Array<{ binding: number, resource: object }>} entries
     * @param {{ label?: string, groupIndex?: number }} [options]
     * @returns {object | null}
     */
    getOrCreateBindGroup(layout, entries, options) {
      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        !layout ||
        !Array.isArray(entries)
      ) {
        this._stats.bindGroupErrors += 1;
        return null;
      }

      const label = options && typeof options.label === 'string' ? options.label : null;
      const groupIndex =
        options && Number.isInteger(options.groupIndex) && options.groupIndex >= 0
          ? options.groupIndex
          : -1;
      return this._getOrCreateBindGroupStructured(BG_OWNER_LAYOUT, layout, groupIndex, entries, label);
    }

    /**
     * Look up or create a cached GPUBindGroup keyed by `(pipeline, groupIndex, entryResources)`.
     * Avoids calling `pipeline.getBindGroupLayout(groupIndex)` and `device.createBindGroup()`
     * on cache hits across draws and compute dispatches.
     *
     * @param {object} pipeline GPURenderPipeline or GPUComputePipeline
     * @param {number} groupIndex
     * @param {Array<{ binding: number, resource: object }>} entries
     * @param {{ label?: string }} [options]
     * @returns {object | null}
     */
    getOrCreateBindGroupForPipeline(pipeline, groupIndex, entries, options) {
      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        !pipeline ||
        typeof pipeline.getBindGroupLayout !== 'function' ||
        !Number.isInteger(groupIndex) ||
        groupIndex < 0 ||
        !Array.isArray(entries)
      ) {
        this._stats.bindGroupErrors += 1;
        return null;
      }

      const label = options && typeof options.label === 'string' ? options.label : null;
      return this._getOrCreateBindGroupStructured(
        BG_OWNER_PIPELINE,
        pipeline,
        groupIndex,
        entries,
        label
      );
    }

    setDevice(device) {
      if (this._device !== device) {
        this._destroySyntheticResources();
        if (this._ownsIndirectClampExecutor && this._indirectClampExecutor) {
          try {
            this._indirectClampExecutor.destroy();
          } catch (_err) {
            // Ignore cleanup errors on previous device.
          }
          this._indirectClampExecutor = null;
          this._ownsIndirectClampExecutor = false;
        }
        this._fallbackPipelineCache.clear();
        this._fallbackShaderModuleCache.clear();
        this._clearBgCacheInternal();
        this._wgslHashCache.clear();
        this._textureViewCache = new WeakMap();
        this._bindGroupLayoutCache = new WeakMap();
        this._clearBoundPassState();
      }
      this._device = device !== undefined ? device : null;
    }

    setShaderFallbackRouter(router) {
      this._shaderFallbackRouter = router !== undefined ? router : null;
    }

    setFormat(format) {
      if (typeof format === 'string' && format.length > 0) {
        this._format = format;
      }
    }

    /**
     * Check whether the active WebGPU device supports `indirect-first-instance`.
     */
    supportsIndirectFirstInstance() {
      if (this._indirectFirstInstanceOverride !== null) {
        return this._indirectFirstInstanceOverride;
      }
      return Boolean(
        this._device &&
          this._device.features &&
          typeof this._device.features.has === 'function' &&
          this._device.features.has('indirect-first-instance')
      );
    }

    /**
     * Lazily obtain (or create) the device-scoped `GpuIndirectClampPass` executor used
     * for GPU-clamped `drawIndirect`, `drawIndexedIndirect`, and `dispatchIndirect`.
     */
    getIndirectClampExecutor() {
      if (this._indirectClampExecutor && this._indirectClampExecutor.destroyed !== true) {
        return this._indirectClampExecutor;
      }
      if (
        !this._device ||
        typeof this._device.createShaderModule !== 'function' ||
        typeof this._device.createComputePipeline !== 'function'
      ) {
        return null;
      }
      const mod = resolveIndirectDrawModule();
      if (!mod || typeof mod.GpuIndirectClampPass !== 'function') {
        return null;
      }
      this._indirectClampExecutor = new mod.GpuIndirectClampPass({
        device: this._device,
        indirectFirstInstance: this.supportsIndirectFirstInstance(),
        maxComputeWorkgroupsPerDimension:
          this._maxComputeWorkgroupsOverride !== null
            ? this._maxComputeWorkgroupsOverride
            : undefined,
      });
      this._ownsIndirectClampExecutor = true;
      return this._indirectClampExecutor;
    }

    /**
     * Reset per-frame draw-visibility counters with zero heap allocations
     * (call at the start of each frame before recording render passes).
     */
    beginFrame() {
      this._frameHasNonClearDraw = false;
      this._frameRealPackDraws = 0;
      this._frameFallbackDraws = 0;
      this._frameDraws = 0;
      if (
        this._indirectClampExecutor &&
        this._indirectClampExecutor.destroyed !== true &&
        typeof this._indirectClampExecutor.resetFrame === 'function'
      ) {
        this._indirectClampExecutor.resetFrame();
      }
    }

    /**
     * Return a JSON-serializable summary of draws executed in the current frame
     * for swapchain visible-frame readiness tracking (W251 -> W231).
     *
     * Pass `true` or `{ reuse: true }` to fill and return a single preallocated
     * mutable snapshot object (zero heap allocation), or pass a caller-owned target
     * object to populate in place. Omit arguments for a fresh 4-field snapshot.
     *
     * @param {boolean | { reuse?: boolean } | object} [targetOrOptions]
     * @returns {{
     *   hasNonClearDraw: boolean,
     *   realPackDrawsThisFrame: number,
     *   fallbackDrawsThisFrame: number,
     *   drawsThisFrame: number,
     * }}
     */
    getFrameDrawSummary(targetOrOptions) {
      if (targetOrOptions === true) {
        const out = this._reusableFrameSummary;
        out.hasNonClearDraw = this._frameHasNonClearDraw;
        out.realPackDrawsThisFrame = this._frameRealPackDraws;
        out.fallbackDrawsThisFrame = this._frameFallbackDraws;
        out.drawsThisFrame = this._frameDraws;
        return out;
      }
      if (targetOrOptions !== null && typeof targetOrOptions === 'object') {
        const out =
          targetOrOptions.reuse === true ? this._reusableFrameSummary : targetOrOptions;
        out.hasNonClearDraw = this._frameHasNonClearDraw;
        out.realPackDrawsThisFrame = this._frameRealPackDraws;
        out.fallbackDrawsThisFrame = this._frameFallbackDraws;
        out.drawsThisFrame = this._frameDraws;
        return out;
      }
      return {
        hasNonClearDraw: this._frameHasNonClearDraw,
        realPackDrawsThisFrame: this._frameRealPackDraws,
        fallbackDrawsThisFrame: this._frameFallbackDraws,
        drawsThisFrame: this._frameDraws,
      };
    }

    /**
     * Reset per-pass bound draw state (call at the start of a render pass).
     */
    beginPass(options) {
      this.resetState();
      if (options && typeof options === 'object') {
        if (typeof options.format === 'string' && options.format.length > 0) {
          this._format = options.format;
        }
        if (options.depthStencilFormat !== undefined) {
          this._depthStencilFormat =
            typeof options.depthStencilFormat === 'string' && options.depthStencilFormat.length > 0
              ? options.depthStencilFormat
              : null;
        }
      }
    }

    /**
     * Mark the end of the active render pass so subsequent passes re-bind state cleanly.
     */
    endPass() {
      this._clearBoundPassState();
    }

    /**
     * Clear all bound pipeline, buffer, texture, sampler, and bind-group state.
     */
    resetState() {
      this._activePipeline = null;
      this._activeComputePipeline = null;

      let vbMask = this._stagedVbMask;
      while (vbMask !== 0) {
        const bit = vbMask & -vbMask;
        const slot = 31 - Math.clz32(bit);
        vbMask ^= bit;
        this._stagedVbBuffers[slot] = null;
        this._stagedVbOffsets[slot] = 0;
        this._stagedVbSizes[slot] = null;
        this._stagedVbStrides[slot] = null;
      }
      this._stagedVbMask = 0;
      if (this._stagedVbOverflow !== null && this._stagedVbOverflow.size > 0) {
        this._stagedVbOverflow.clear();
      }

      let bgMask = this._stagedBgMask;
      while (bgMask !== 0) {
        const bit = bgMask & -bgMask;
        const slot = 31 - Math.clz32(bit);
        bgMask ^= bit;
        this._stagedBgRaw[slot] = null;
        this._stagedBgGroups[slot] = null;
        this._stagedBgMeta[slot] = null;
        this._stagedBgHasDyn[slot] = 0;
        this._stagedBgDynSource[slot] = null;
        this._stagedBgDynStart[slot] = 0;
        this._stagedBgDynLength[slot] = -1;
      }
      this._stagedBgMask = 0;
      if (this._stagedBgOverflow !== null && this._stagedBgOverflow.size > 0) {
        this._stagedBgOverflow.clear();
      }

      this._hasStagedIb = false;
      this._stagedIbBuffer = null;
      this._stagedIbFormat = 'uint16';
      this._stagedIbOffset = 0;
      this._stagedIbSize = null;
      this._dirtyIb = false;

      this._clearBoundPassState();
      this._cbvs.clear();
      this._textures.clear();
      this._samplers.clear();
      this._hasStagedViewport = false;
      this._hasStagedScissor = false;
      this._hasStagedBlendConstant = false;
      this._stencilReference = null;
    }

    /**
     * Reset active per-pass state (and optionally telemetry counters `{ stats: true }`
     * and/or device-scoped caches `{ caches: true }`).
     */
    reset(options) {
      this.resetState();
      if (options && typeof options === 'object') {
        if (options.stats === true) {
          this.resetStats();
        }
        if (options.caches === true) {
          this.clearBindGroupCache();
          this._fallbackPipelineCache.clear();
          this._fallbackShaderModuleCache.clear();
          this._wgslHashCache.clear();
        }
      }
    }

    _fastDynamicOffsetsEqual(slot, hasDyn, src, start, explicitLen) {
      if (this._boundBgHasDyn[slot] !== hasDyn) {
        return false;
      }
      if (hasDyn === 0) {
        return true;
      }
      const len = explicitLen >= 0 ? explicitLen : src.length;
      if (this._boundBgDynLen[slot] !== len) {
        return false;
      }
      if (len === 0) {
        return true;
      }
      if (len <= MAX_FAST_DYN_OFFSETS) {
        const snap = this._boundBgDynOffsets;
        const base = slot * MAX_FAST_DYN_OFFSETS;
        for (let i = 0; i < len; i += 1) {
          if (snap[base + i] !== src[start + i]) {
            return false;
          }
        }
        return true;
      }
      const overflow = this._boundBgDynOverflow[slot];
      if (!overflow || overflow.length < len) {
        return false;
      }
      for (let i = 0; i < len; i += 1) {
        if (overflow[i] !== src[start + i]) {
          return false;
        }
      }
      return true;
    }

    _copyFastDynamicOffsets(slot, hasDyn, src, start, explicitLen) {
      this._boundBgHasDyn[slot] = hasDyn;
      if (hasDyn === 0) {
        this._boundBgDynLen[slot] = 0;
        return;
      }
      const len = explicitLen >= 0 ? explicitLen : src.length;
      this._boundBgDynLen[slot] = len;
      if (len === 0) {
        return;
      }
      if (len <= MAX_FAST_DYN_OFFSETS) {
        const snap = this._boundBgDynOffsets;
        const base = slot * MAX_FAST_DYN_OFFSETS;
        for (let i = 0; i < len; i += 1) {
          snap[base + i] = src[start + i];
        }
        return;
      }
      let overflow = this._boundBgDynOverflow[slot];
      if (!overflow || overflow.length < len) {
        overflow = new Float64Array(len);
        this._boundBgDynOverflow[slot] = overflow;
      }
      for (let i = 0; i < len; i += 1) {
        overflow[i] = src[start + i];
      }
    }

    _bindPipelineOnPass(pass, gpuPipeline) {
      this._syncPassTracking(pass);
      if (
        this._elideRedundantPipelines &&
        this._boundPipeline === gpuPipeline &&
        gpuPipeline !== null &&
        gpuPipeline !== undefined
      ) {
        this._stats.redundantPipelineSkips += 1;
        this._stats.elidedPipelineSets += 1;
        this._stats.totalElidedStateCalls += 1;
        this._totalStateCalls += 1;
        return false;
      }
      pass.setPipeline(gpuPipeline);
      this._boundPipeline = gpuPipeline;
      this._totalStateCalls += 1;
      return true;
    }

    _bindFastBindGroupOnPassSynced(pass, slot, bit, bindGroup, hasDyn, src, start, explicitLen) {
      if (
        this._elideRedundantBindGroups &&
        (this._boundBgMask & bit) !== 0 &&
        bindGroup !== null &&
        bindGroup !== undefined &&
        bindGroup.destroyed !== true &&
        this._boundBgGroups[slot] === bindGroup &&
        this._fastDynamicOffsetsEqual(slot, hasDyn, src, start, explicitLen)
      ) {
        this._dirtyBgMask &= ~bit;
        return false;
      }
      if (hasDyn === 1) {
        const len = explicitLen >= 0 ? explicitLen : src.length;
        if (start === 0 && len === src.length) {
          pass.setBindGroup(slot, bindGroup, src);
        } else {
          const slice =
            typeof src.subarray === 'function'
              ? src.subarray(start, start + len)
              : Array.prototype.slice.call(src, start, start + len);
          pass.setBindGroup(slot, bindGroup, slice);
        }
      } else {
        pass.setBindGroup(slot, bindGroup);
      }
      this._totalStateCalls += 1;
      this._boundBgMask |= bit;
      this._boundBgGroups[slot] = bindGroup;
      this._copyFastDynamicOffsets(slot, hasDyn, src, start, explicitLen);
      if (
        (this._stagedBgMask & bit) !== 0 &&
        this._stagedBgGroups[slot] === bindGroup &&
        this._stagedBgHasDyn[slot] === hasDyn
      ) {
        this._dirtyBgMask &= ~bit;
      } else if ((this._stagedBgMask & bit) !== 0) {
        this._dirtyBgMask |= bit;
      }
      return true;
    }

    _bindBindGroupOnPass(pass, groupIndex, bindGroup, dynamicOffsets) {
      this._syncPassTracking(pass);
      const hasDyn = dynamicOffsets !== null && dynamicOffsets !== undefined ? 1 : 0;
      if (groupIndex >= 0 && groupIndex < MAX_FAST_BG_SLOTS) {
        const bit = 1 << groupIndex;
        const bound = this._bindFastBindGroupOnPassSynced(
          pass,
          groupIndex,
          bit,
          bindGroup,
          hasDyn,
          hasDyn === 1 ? dynamicOffsets : null,
          0,
          -1
        );
        if (!bound) {
          this._stats.redundantBindGroupSkips += 1;
          this._stats.elidedBindGroupSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
        }
        return bound;
      }

      const offsets = hasDyn === 1 ? dynamicOffsets : null;
      if (
        this._elideRedundantBindGroups &&
        this._boundBgOverflow !== null &&
        bindGroup !== null &&
        bindGroup !== undefined &&
        bindGroup.destroyed !== true
      ) {
        const bound = this._boundBgOverflow.get(groupIndex);
        if (bound && bound.bindGroup === bindGroup && dynamicOffsetsEqual(bound, offsets)) {
          this._stats.redundantBindGroupSkips += 1;
          this._stats.elidedBindGroupSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
          return false;
        }
      }
      if (offsets !== null) {
        pass.setBindGroup(groupIndex, bindGroup, offsets);
      } else {
        pass.setBindGroup(groupIndex, bindGroup);
      }
      this._totalStateCalls += 1;
      if (this._boundBgOverflow === null) {
        this._boundBgOverflow = new Map();
      }
      let record = this._boundBgOverflow.get(groupIndex);
      if (!record) {
        record = {
          bindGroup,
          hasDynamicOffsets: false,
          dynamicOffsetsLength: 0,
          dynamicOffsetsSnapshot: null,
        };
        this._boundBgOverflow.set(groupIndex, record);
      } else {
        record.bindGroup = bindGroup;
      }
      copyDynamicOffsetsInto(record, offsets);
      return true;
    }

    _bindFastVertexBufferOnPassSynced(pass, slot, bit, buffer, offset, size) {
      const normOffset = offset || 0;
      const hasSize = size !== undefined && size !== null;
      const normSize = hasSize ? size : null;
      if (
        this._elideRedundantVertexBuffers &&
        (this._boundVbMask & bit) !== 0 &&
        buffer !== null &&
        buffer !== undefined &&
        buffer.destroyed !== true &&
        this._boundVbBuffers[slot] === buffer &&
        this._boundVbOffsets[slot] === normOffset &&
        this._boundVbSizes[slot] === normSize
      ) {
        this._dirtyVbMask &= ~bit;
        return false;
      }
      if (hasSize) {
        pass.setVertexBuffer(slot, buffer, normOffset, size);
      } else {
        pass.setVertexBuffer(slot, buffer, normOffset);
      }
      this._totalStateCalls += 1;
      this._boundVbMask |= bit;
      this._boundVbBuffers[slot] = buffer;
      this._boundVbOffsets[slot] = normOffset;
      this._boundVbSizes[slot] = normSize;
      if (
        (this._stagedVbMask & bit) !== 0 &&
        this._stagedVbBuffers[slot] === buffer &&
        (this._stagedVbOffsets[slot] || 0) === normOffset &&
        this._stagedVbSizes[slot] === normSize
      ) {
        this._dirtyVbMask &= ~bit;
      } else if ((this._stagedVbMask & bit) !== 0) {
        this._dirtyVbMask |= bit;
      }
      return true;
    }

    _bindVertexBufferOnPass(pass, slot, buffer, offset, size) {
      this._syncPassTracking(pass);
      if (slot >= 0 && slot < MAX_FAST_VB_SLOTS) {
        const bit = 1 << slot;
        const bound = this._bindFastVertexBufferOnPassSynced(pass, slot, bit, buffer, offset, size);
        if (!bound) {
          this._stats.redundantVertexBufferSkips += 1;
          this._stats.elidedVertexBufferSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
        }
        return bound;
      }

      const normOffset = offset || 0;
      const hasSize = size !== undefined && size !== null;
      const normSize = hasSize ? size : null;
      if (
        this._elideRedundantVertexBuffers &&
        this._boundVbOverflow !== null &&
        buffer !== null &&
        buffer !== undefined &&
        buffer.destroyed !== true
      ) {
        const bound = this._boundVbOverflow.get(slot);
        if (
          bound &&
          bound.buffer === buffer &&
          bound.offset === normOffset &&
          bound.size === normSize
        ) {
          this._stats.redundantVertexBufferSkips += 1;
          this._stats.elidedVertexBufferSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
          return false;
        }
      }
      if (hasSize) {
        pass.setVertexBuffer(slot, buffer, normOffset, size);
      } else {
        pass.setVertexBuffer(slot, buffer, normOffset);
      }
      this._totalStateCalls += 1;
      if (this._boundVbOverflow === null) {
        this._boundVbOverflow = new Map();
      }
      let record = this._boundVbOverflow.get(slot);
      if (!record) {
        record = { buffer, offset: normOffset, size: normSize };
        this._boundVbOverflow.set(slot, record);
      } else {
        record.buffer = buffer;
        record.offset = normOffset;
        record.size = normSize;
      }
      return true;
    }

    _bindIndexBufferOnPass(pass, buffer, indexFormat, offset, size) {
      this._syncPassTracking(pass);
      const normOffset = offset || 0;
      const hasSize = size !== undefined && size !== null;
      const normSize = hasSize ? size : null;
      if (
        this._elideRedundantIndexBuffers &&
        this._hasBoundIb &&
        this._boundIbBuffer === buffer &&
        this._boundIbFormat === indexFormat &&
        this._boundIbOffset === normOffset &&
        this._boundIbSize === normSize &&
        buffer !== null &&
        buffer !== undefined &&
        buffer.destroyed !== true
      ) {
        this._dirtyIb = false;
        this._stats.redundantIndexBufferSkips += 1;
        this._stats.elidedIndexBufferSets += 1;
        this._stats.totalElidedStateCalls += 1;
        this._totalStateCalls += 1;
        return false;
      }
      if (hasSize) {
        pass.setIndexBuffer(buffer, indexFormat, normOffset, size);
      } else {
        pass.setIndexBuffer(buffer, indexFormat, normOffset);
      }
      this._totalStateCalls += 1;
      this._hasBoundIb = true;
      this._boundIbBuffer = buffer;
      this._boundIbFormat = indexFormat;
      this._boundIbOffset = normOffset;
      this._boundIbSize = normSize;
      this._dirtyIb = false;
      return true;
    }

    _bindViewportOnPass(pass, vp) {
      this._syncPassTracking(pass);
      if (typeof pass.setViewport !== 'function') {
        return BIND_STATUS_UNSUPPORTED;
      }
      if (this._elideRedundantViewportScissor && this._hasBoundViewport) {
        const bound = this._boundViewportData;
        if (
          bound.x === vp.x &&
          bound.y === vp.y &&
          bound.width === vp.width &&
          bound.height === vp.height &&
          bound.minDepth === vp.minDepth &&
          bound.maxDepth === vp.maxDepth
        ) {
          this._stats.elidedViewportScissorSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
          return BIND_STATUS_ELIDED;
        }
      }
      pass.setViewport(vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth);
      this._totalStateCalls += 1;
      const bound = this._boundViewportData;
      bound.x = vp.x;
      bound.y = vp.y;
      bound.width = vp.width;
      bound.height = vp.height;
      bound.minDepth = vp.minDepth;
      bound.maxDepth = vp.maxDepth;
      this._hasBoundViewport = true;
      return BIND_STATUS_EMITTED;
    }

    _bindScissorRectOnPass(pass, sc) {
      this._syncPassTracking(pass);
      if (typeof pass.setScissorRect !== 'function') {
        return BIND_STATUS_UNSUPPORTED;
      }
      if (this._elideRedundantViewportScissor && this._hasBoundScissor) {
        const bound = this._boundScissorData;
        if (
          bound.x === sc.x &&
          bound.y === sc.y &&
          bound.width === sc.width &&
          bound.height === sc.height
        ) {
          this._stats.elidedViewportScissorSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
          return BIND_STATUS_ELIDED;
        }
      }
      pass.setScissorRect(sc.x, sc.y, sc.width, sc.height);
      this._totalStateCalls += 1;
      const bound = this._boundScissorData;
      bound.x = sc.x;
      bound.y = sc.y;
      bound.width = sc.width;
      bound.height = sc.height;
      this._hasBoundScissor = true;
      return BIND_STATUS_EMITTED;
    }

    _bindBlendConstantOnPass(pass, bc) {
      this._syncPassTracking(pass);
      if (typeof pass.setBlendConstant !== 'function') {
        return BIND_STATUS_UNSUPPORTED;
      }
      if (this._elideRedundantBlendStencil && this._hasBoundBlendConstant) {
        const bound = this._boundBlendConstantData;
        if (bound.r === bc.r && bound.g === bc.g && bound.b === bc.b && bound.a === bc.a) {
          this._stats.elidedBlendConstantSets += 1;
          this._stats.totalElidedStateCalls += 1;
          this._totalStateCalls += 1;
          return BIND_STATUS_ELIDED;
        }
      }
      pass.setBlendConstant({ r: bc.r, g: bc.g, b: bc.b, a: bc.a });
      this._totalStateCalls += 1;
      const bound = this._boundBlendConstantData;
      bound.r = bc.r;
      bound.g = bc.g;
      bound.b = bc.b;
      bound.a = bc.a;
      this._hasBoundBlendConstant = true;
      return BIND_STATUS_EMITTED;
    }

    _bindStencilReferenceOnPass(pass, ref) {
      this._syncPassTracking(pass);
      if (typeof pass.setStencilReference !== 'function') {
        return BIND_STATUS_UNSUPPORTED;
      }
      if (this._elideRedundantBlendStencil && this._boundStencilReference === ref) {
        this._stats.elidedStencilReferenceSets += 1;
        this._stats.totalElidedStateCalls += 1;
        this._totalStateCalls += 1;
        return BIND_STATUS_ELIDED;
      }
      pass.setStencilReference(ref);
      this._boundStencilReference = ref;
      this._totalStateCalls += 1;
      return BIND_STATUS_EMITTED;
    }

    _bindPassDynamicState(pass, args, result) {
      if (args === null) {
        if (this._hasStagedViewport) {
          if (this._bindViewportOnPass(pass, this._stagedViewport) === BIND_STATUS_ELIDED) {
            result.viewportElided = true;
          }
        }
        if (this._hasStagedScissor) {
          if (this._bindScissorRectOnPass(pass, this._stagedScissor) === BIND_STATUS_ELIDED) {
            result.scissorElided = true;
          }
        }
        if (this._hasStagedBlendConstant) {
          if (
            this._bindBlendConstantOnPass(pass, this._stagedBlendConstant) === BIND_STATUS_ELIDED
          ) {
            result.blendConstantElided = true;
          }
        }
        if (this._stencilReference !== null) {
          if (
            this._bindStencilReferenceOnPass(pass, this._stencilReference) === BIND_STATUS_ELIDED
          ) {
            result.stencilReferenceElided = true;
          }
        }
        return;
      }

      if (args.viewport !== undefined) {
        if (
          args.viewport !== null &&
          normalizeViewportArgsInto(this._tmpViewport, args.viewport)
        ) {
          if (this._bindViewportOnPass(pass, this._tmpViewport) === BIND_STATUS_ELIDED) {
            result.viewportElided = true;
          }
        }
      } else if (this._hasStagedViewport) {
        if (this._bindViewportOnPass(pass, this._stagedViewport) === BIND_STATUS_ELIDED) {
          result.viewportElided = true;
        }
      }

      const rawScissor =
        args.scissorRect !== undefined
          ? args.scissorRect
          : args.scissor !== undefined
            ? args.scissor
            : undefined;
      if (rawScissor !== undefined) {
        if (rawScissor !== null && normalizeScissorRectArgsInto(this._tmpScissor, rawScissor)) {
          if (this._bindScissorRectOnPass(pass, this._tmpScissor) === BIND_STATUS_ELIDED) {
            result.scissorElided = true;
          }
        }
      } else if (this._hasStagedScissor) {
        if (this._bindScissorRectOnPass(pass, this._stagedScissor) === BIND_STATUS_ELIDED) {
          result.scissorElided = true;
        }
      }

      if (args.blendConstant !== undefined) {
        if (
          args.blendConstant !== null &&
          normalizeBlendConstantArgsInto(this._tmpBlendConstant, args.blendConstant)
        ) {
          if (
            this._bindBlendConstantOnPass(pass, this._tmpBlendConstant) === BIND_STATUS_ELIDED
          ) {
            result.blendConstantElided = true;
          }
        }
      } else if (this._hasStagedBlendConstant) {
        if (this._bindBlendConstantOnPass(pass, this._stagedBlendConstant) === BIND_STATUS_ELIDED) {
          result.blendConstantElided = true;
        }
      }

      const sr =
        args.stencilReference !== undefined
          ? args.stencilReference === null
            ? null
            : normalizeStencilReference(args.stencilReference)
          : this._stencilReference;
      if (sr !== null) {
        if (this._bindStencilReferenceOnPass(pass, sr) === BIND_STATUS_ELIDED) {
          result.stencilReferenceElided = true;
        }
      }
    }

    setPipeline(pipelineOrDesc) {
      this._activePipeline = pipelineOrDesc !== undefined ? pipelineOrDesc : null;
    }

    setComputePipeline(pipelineOrDesc) {
      this._activeComputePipeline = pipelineOrDesc !== undefined ? pipelineOrDesc : null;
    }

    setVertexBuffer(slot, buffer, offsetOrOptions, size) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.invalidVertexBufferGuards += 1;
        this._stats.skippedVertexBufferBindings += 1;
        return false;
      }
      if (slot < MAX_FAST_VB_SLOTS) {
        const bit = 1 << slot;
        if (buffer === null || buffer === undefined) {
          this._stagedVbMask &= ~bit;
          this._dirtyVbMask &= ~bit;
          this._boundVbMask &= ~bit;
          this._stagedVbBuffers[slot] = null;
          this._stagedVbOffsets[slot] = 0;
          this._stagedVbSizes[slot] = null;
          this._stagedVbStrides[slot] = null;
          this._boundVbBuffers[slot] = null;
          this._boundVbOffsets[slot] = 0;
          this._boundVbSizes[slot] = null;
          return true;
        }
        let offset = 0;
        let explicitSize = size !== undefined ? size : null;
        let stride = null;

        if (typeof offsetOrOptions === 'number') {
          offset = offsetOrOptions;
        } else if (typeof offsetOrOptions === 'object' && offsetOrOptions !== null) {
          offset = offsetOrOptions.offset !== undefined ? offsetOrOptions.offset : 0;
          if (offsetOrOptions.size !== undefined) {
            explicitSize = offsetOrOptions.size;
          }
          if (offsetOrOptions.stride !== undefined) {
            stride = offsetOrOptions.stride;
          } else if (offsetOrOptions.arrayStride !== undefined) {
            stride = offsetOrOptions.arrayStride;
          }
        }

        if (stride === null && typeof buffer === 'object') {
          if (typeof buffer.stride === 'number') {
            stride = buffer.stride;
          } else if (typeof buffer.arrayStride === 'number') {
            stride = buffer.arrayStride;
          }
        }

        const storedOffset = typeof offset === 'number' ? offset : NaN;
        const normOffset = storedOffset || 0;
        const normSize = explicitSize !== undefined && explicitSize !== null ? explicitSize : null;
        this._stagedVbBuffers[slot] = buffer;
        this._stagedVbOffsets[slot] = storedOffset;
        this._stagedVbSizes[slot] = explicitSize;
        this._stagedVbStrides[slot] = stride;
        this._stagedVbMask |= bit;

        if (
          (this._boundVbMask & bit) !== 0 &&
          this._boundVbBuffers[slot] === buffer &&
          this._boundVbOffsets[slot] === normOffset &&
          this._boundVbSizes[slot] === normSize &&
          Number.isInteger(storedOffset) &&
          storedOffset >= 0 &&
          storedOffset % VERTEX_OFFSET_ALIGNMENT === 0
        ) {
          this._dirtyVbMask &= ~bit;
        } else {
          this._dirtyVbMask |= bit;
        }
        return true;
      }

      if (buffer === null || buffer === undefined) {
        if (this._stagedVbOverflow !== null) {
          this._stagedVbOverflow.delete(slot);
        }
        if (this._boundVbOverflow !== null) {
          this._boundVbOverflow.delete(slot);
        }
        return true;
      }
      let offset = 0;
      let explicitSize = size !== undefined ? size : null;
      let stride = null;

      if (typeof offsetOrOptions === 'object' && offsetOrOptions !== null) {
        offset = offsetOrOptions.offset !== undefined ? offsetOrOptions.offset : 0;
        if (offsetOrOptions.size !== undefined) {
          explicitSize = offsetOrOptions.size;
        }
        if (offsetOrOptions.stride !== undefined) {
          stride = offsetOrOptions.stride;
        } else if (offsetOrOptions.arrayStride !== undefined) {
          stride = offsetOrOptions.arrayStride;
        }
      } else if (typeof offsetOrOptions === 'number') {
        offset = offsetOrOptions;
      }

      if (stride === null && typeof buffer === 'object') {
        if (typeof buffer.stride === 'number') {
          stride = buffer.stride;
        } else if (typeof buffer.arrayStride === 'number') {
          stride = buffer.arrayStride;
        }
      }

      if (this._stagedVbOverflow === null) {
        this._stagedVbOverflow = new Map();
      }
      this._stagedVbOverflow.set(slot, {
        slot,
        buffer,
        offset,
        size: explicitSize,
        stride,
      });
      return true;
    }

    setIndexBuffer(buffer, indexFormat = 'uint16', offset = 0, size = null) {
      if (buffer === null || buffer === undefined) {
        this._hasStagedIb = false;
        this._stagedIbBuffer = null;
        this._stagedIbFormat = 'uint16';
        this._stagedIbOffset = 0;
        this._stagedIbSize = null;
        this._hasBoundIb = false;
        this._boundIbBuffer = null;
        this._boundIbFormat = null;
        this._boundIbOffset = 0;
        this._boundIbSize = null;
        this._dirtyIb = false;
        return true;
      }
      this._hasStagedIb = true;
      this._stagedIbBuffer = buffer;
      this._stagedIbFormat = indexFormat;
      this._stagedIbOffset = offset;
      this._stagedIbSize = size;
      const normOffset = offset || 0;
      const normSize = size !== undefined && size !== null ? size : null;
      this._dirtyIb = !(
        this._hasBoundIb &&
        this._boundIbBuffer === buffer &&
        this._boundIbFormat === indexFormat &&
        this._boundIbOffset === normOffset &&
        this._boundIbSize === normSize
      );
      return true;
    }

    setBindGroup(
      groupIndex,
      bindGroup,
      dynamicOffsetsOrMeta,
      dynamicOffsetsDataStart,
      dynamicOffsetsDataLength
    ) {
      if (!Number.isInteger(groupIndex) || groupIndex < 0) {
        this._stats.bindGroupGuards += 1;
        this._stats.skippedBindGroupBindings += 1;
        return false;
      }
      if (groupIndex < MAX_FAST_BG_SLOTS) {
        const bit = 1 << groupIndex;
        if (bindGroup === null || bindGroup === undefined) {
          this._stagedBgMask &= ~bit;
          this._dirtyBgMask &= ~bit;
          this._boundBgMask &= ~bit;
          this._stagedBgRaw[groupIndex] = null;
          this._stagedBgGroups[groupIndex] = null;
          this._stagedBgMeta[groupIndex] = null;
          this._stagedBgHasDyn[groupIndex] = 0;
          this._stagedBgDynSource[groupIndex] = null;
          this._stagedBgDynStart[groupIndex] = 0;
          this._stagedBgDynLength[groupIndex] = -1;
          this._boundBgGroups[groupIndex] = null;
          this._boundBgHasDyn[groupIndex] = 0;
          this._boundBgDynLen[groupIndex] = 0;
          return true;
        }

        let hasDyn = 0;
        let dynSrc = null;
        let dynStart = 0;
        let dynLen = -1;
        let meta = null;

        if (dynamicOffsetsOrMeta !== undefined && dynamicOffsetsOrMeta !== null) {
          if (Array.isArray(dynamicOffsetsOrMeta) || ArrayBuffer.isView(dynamicOffsetsOrMeta)) {
            hasDyn = 1;
            dynSrc = dynamicOffsetsOrMeta;
            if (
              ArrayBuffer.isView(dynamicOffsetsOrMeta) &&
              typeof dynamicOffsetsDataStart === 'number' &&
              Number.isInteger(dynamicOffsetsDataStart) &&
              dynamicOffsetsDataStart >= 0
            ) {
              const srcLen = dynamicOffsetsOrMeta.length;
              const clampedStart =
                dynamicOffsetsDataStart > srcLen ? srcLen : dynamicOffsetsDataStart;
              const rawEnd =
                typeof dynamicOffsetsDataLength === 'number' &&
                Number.isInteger(dynamicOffsetsDataLength) &&
                dynamicOffsetsDataLength >= 0
                  ? clampedStart + dynamicOffsetsDataLength
                  : srcLen;
              const clampedEnd = rawEnd > srcLen ? srcLen : rawEnd;
              dynStart = clampedStart;
              dynLen = clampedEnd - clampedStart;
            }
          } else if (typeof dynamicOffsetsOrMeta === 'object') {
            meta = dynamicOffsetsOrMeta;
            if (
              Array.isArray(dynamicOffsetsOrMeta.dynamicOffsets) ||
              ArrayBuffer.isView(dynamicOffsetsOrMeta.dynamicOffsets)
            ) {
              hasDyn = 1;
              dynSrc = dynamicOffsetsOrMeta.dynamicOffsets;
            }
          }
        }

        const actualBg = bindGroup.bindGroup !== undefined ? bindGroup.bindGroup : bindGroup;
        this._stagedBgRaw[groupIndex] = bindGroup;
        this._stagedBgGroups[groupIndex] = actualBg;
        this._stagedBgMeta[groupIndex] = meta;
        this._stagedBgHasDyn[groupIndex] = hasDyn;
        this._stagedBgDynSource[groupIndex] = dynSrc;
        this._stagedBgDynStart[groupIndex] = dynStart;
        this._stagedBgDynLength[groupIndex] = dynLen;
        this._stagedBgMask |= bit;

        if (
          (this._boundBgMask & bit) !== 0 &&
          this._boundBgGroups[groupIndex] === actualBg &&
          this._fastDynamicOffsetsEqual(groupIndex, hasDyn, dynSrc, dynStart, dynLen)
        ) {
          this._dirtyBgMask &= ~bit;
        } else {
          this._dirtyBgMask |= bit;
        }
        return true;
      }

      if (bindGroup === null || bindGroup === undefined) {
        if (this._stagedBgOverflow !== null) {
          this._stagedBgOverflow.delete(groupIndex);
        }
        if (this._boundBgOverflow !== null) {
          this._boundBgOverflow.delete(groupIndex);
        }
        return true;
      }
      let dynamicOffsets = null;
      let meta = null;
      if (Array.isArray(dynamicOffsetsOrMeta) || ArrayBuffer.isView(dynamicOffsetsOrMeta)) {
        if (
          ArrayBuffer.isView(dynamicOffsetsOrMeta) &&
          typeof dynamicOffsetsDataStart === 'number' &&
          Number.isInteger(dynamicOffsetsDataStart) &&
          dynamicOffsetsDataStart >= 0
        ) {
          const end =
            typeof dynamicOffsetsDataLength === 'number' &&
            Number.isInteger(dynamicOffsetsDataLength) &&
            dynamicOffsetsDataLength >= 0
              ? dynamicOffsetsDataStart + dynamicOffsetsDataLength
              : dynamicOffsetsOrMeta.length;
          dynamicOffsets =
            typeof dynamicOffsetsOrMeta.subarray === 'function'
              ? dynamicOffsetsOrMeta.subarray(dynamicOffsetsDataStart, end)
              : Array.prototype.slice.call(dynamicOffsetsOrMeta, dynamicOffsetsDataStart, end);
        } else {
          dynamicOffsets = dynamicOffsetsOrMeta;
        }
      } else if (dynamicOffsetsOrMeta && typeof dynamicOffsetsOrMeta === 'object') {
        meta = dynamicOffsetsOrMeta;
        if (
          Array.isArray(dynamicOffsetsOrMeta.dynamicOffsets) ||
          ArrayBuffer.isView(dynamicOffsetsOrMeta.dynamicOffsets)
        ) {
          dynamicOffsets = dynamicOffsetsOrMeta.dynamicOffsets;
        }
      }

      if (this._stagedBgOverflow === null) {
        this._stagedBgOverflow = new Map();
      }
      this._stagedBgOverflow.set(groupIndex, {
        groupIndex,
        bindGroup: bindGroup.bindGroup !== undefined ? bindGroup.bindGroup : bindGroup,
        dynamicOffsets,
        entries:
          (meta && meta.entries) ||
          (bindGroup && bindGroup.entries) ||
          (bindGroup && bindGroup.desc && bindGroup.desc.entries) ||
          null,
        hasCbv: Boolean((meta && meta.hasCbv) || (bindGroup && bindGroup.hasCbv)),
        hasTexture: Boolean((meta && meta.hasTexture) || (bindGroup && bindGroup.hasTexture)),
        hasSampler: Boolean((meta && meta.hasSampler) || (bindGroup && bindGroup.hasSampler)),
        cbvBuffer: (meta && meta.cbvBuffer) || (bindGroup && bindGroup.cbvBuffer) || null,
        textureView: (meta && meta.textureView) || (bindGroup && bindGroup.textureView) || null,
        sampler: (meta && meta.sampler) || (bindGroup && bindGroup.sampler) || null,
      });
      return true;
    }

    setConstantBuffer(slot, buffer, options) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (buffer === null || buffer === undefined) {
        this._cbvs.delete(slot);
        return true;
      }
      const opts = options && typeof options === 'object' ? options : {};
      this._cbvs.set(slot, {
        slot,
        buffer,
        offset: opts.offset !== undefined ? opts.offset : 0,
        size: opts.size !== undefined ? opts.size : null,
      });
      return true;
    }

    setCbv(slot, buffer, options) {
      return this.setConstantBuffer(slot, buffer, options);
    }

    setTexture(slot, textureOrView) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (textureOrView === null || textureOrView === undefined) {
        this._textures.delete(slot);
        return true;
      }
      this._textures.set(slot, textureOrView);
      return true;
    }

    setSrv(slot, textureOrView) {
      return this.setTexture(slot, textureOrView);
    }

    setSampler(slot, sampler) {
      if (!Number.isInteger(slot) || slot < 0) {
        this._stats.bindGroupGuards += 1;
        return false;
      }
      if (sampler === null || sampler === undefined) {
        this._samplers.delete(slot);
        return true;
      }
      this._samplers.set(slot, sampler);
      return true;
    }

    /**
     * Stage viewport state for subsequent draws in the active pass, or emit immediately with
     * elision when a `GPURenderPassEncoder` is supplied as the first argument.
     *
     * Overloads:
     * - `setViewport(x, y, width, height, minDepth = 0, maxDepth = 1)`
     * - `setViewport({ x, y, width, height, minDepth?, maxDepth? })`
     * - `setViewport(pass, x, y, width, height, minDepth = 0, maxDepth = 1)`
     * - `setViewport(pass, { x, y, width, height, minDepth?, maxDepth? })`
     * - `setViewport(null)` to clear staged & bound viewport state
     */
    setViewport(xOrPass, y, width, height, minDepth, maxDepth) {
      if (xOrPass === null || xOrPass === undefined) {
        this._hasStagedViewport = false;
        this._hasBoundViewport = false;
        return true;
      }
      if (isRenderPassObject(xOrPass, 'setViewport')) {
        if (this._destroyed) {
          return false;
        }
        if (
          !normalizeViewportArgsInto(
            this._tmpViewport,
            y,
            width,
            height,
            minDepth,
            maxDepth,
            arguments[6]
          )
        ) {
          return false;
        }
        this._hasStagedViewport = false;
        try {
          return this._bindViewportOnPass(xOrPass, this._tmpViewport) !== BIND_STATUS_UNSUPPORTED;
        } catch (err) {
          this._stats.passErrors += 1;
          this._warn('setViewport failed on pass', {
            error: String(err && err.message ? err.message : err),
          });
          return false;
        }
      }
      if (
        !normalizeViewportArgsInto(this._tmpViewport, xOrPass, y, width, height, minDepth, maxDepth)
      ) {
        return false;
      }
      const dst = this._stagedViewport;
      const src = this._tmpViewport;
      dst.x = src.x;
      dst.y = src.y;
      dst.width = src.width;
      dst.height = src.height;
      dst.minDepth = src.minDepth;
      dst.maxDepth = src.maxDepth;
      this._hasStagedViewport = true;
      return true;
    }

    /**
     * Stage scissor rect state for subsequent draws in the active pass, or emit immediately with
     * elision when a `GPURenderPassEncoder` is supplied as the first argument.
     *
     * Overloads:
     * - `setScissorRect(x, y, width, height)`
     * - `setScissorRect({ x, y, width, height })`
     * - `setScissorRect(pass, x, y, width, height)`
     * - `setScissorRect(pass, { x, y, width, height })`
     * - `setScissorRect(null)` to clear staged & bound scissor state
     */
    setScissorRect(xOrPass, y, width, height) {
      if (xOrPass === null || xOrPass === undefined) {
        this._hasStagedScissor = false;
        this._hasBoundScissor = false;
        return true;
      }
      if (isRenderPassObject(xOrPass, 'setScissorRect')) {
        if (this._destroyed) {
          return false;
        }
        if (!normalizeScissorRectArgsInto(this._tmpScissor, y, width, height, arguments[4])) {
          return false;
        }
        this._hasStagedScissor = false;
        try {
          return this._bindScissorRectOnPass(xOrPass, this._tmpScissor) !== BIND_STATUS_UNSUPPORTED;
        } catch (err) {
          this._stats.passErrors += 1;
          this._warn('setScissorRect failed on pass', {
            error: String(err && err.message ? err.message : err),
          });
          return false;
        }
      }
      if (!normalizeScissorRectArgsInto(this._tmpScissor, xOrPass, y, width, height)) {
        return false;
      }
      const dst = this._stagedScissor;
      const src = this._tmpScissor;
      dst.x = src.x;
      dst.y = src.y;
      dst.width = src.width;
      dst.height = src.height;
      this._hasStagedScissor = true;
      return true;
    }

    /**
     * Stage blend constant color for subsequent draws in the active pass, or emit immediately
     * with elision when a `GPURenderPassEncoder` is supplied as the first argument.
     */
    setBlendConstant(colorOrPass, g, b, a) {
      if (colorOrPass === null || colorOrPass === undefined) {
        this._hasStagedBlendConstant = false;
        this._hasBoundBlendConstant = false;
        return true;
      }
      if (isRenderPassObject(colorOrPass, 'setBlendConstant')) {
        if (this._destroyed) {
          return false;
        }
        if (!normalizeBlendConstantArgsInto(this._tmpBlendConstant, g, b, a, arguments[4])) {
          return false;
        }
        this._hasStagedBlendConstant = false;
        try {
          return (
            this._bindBlendConstantOnPass(colorOrPass, this._tmpBlendConstant) !==
            BIND_STATUS_UNSUPPORTED
          );
        } catch (err) {
          this._stats.passErrors += 1;
          this._warn('setBlendConstant failed on pass', {
            error: String(err && err.message ? err.message : err),
          });
          return false;
        }
      }
      if (!normalizeBlendConstantArgsInto(this._tmpBlendConstant, colorOrPass, g, b, a)) {
        return false;
      }
      const dst = this._stagedBlendConstant;
      const src = this._tmpBlendConstant;
      dst.r = src.r;
      dst.g = src.g;
      dst.b = src.b;
      dst.a = src.a;
      this._hasStagedBlendConstant = true;
      return true;
    }

    /**
     * Stage stencil reference value for subsequent draws in the active pass, or emit immediately
     * with elision when a `GPURenderPassEncoder` is supplied as the first argument.
     */
    setStencilReference(refOrPass, reference) {
      if (refOrPass === null || refOrPass === undefined) {
        this._stencilReference = null;
        this._boundStencilReference = null;
        return true;
      }
      if (isRenderPassObject(refOrPass, 'setStencilReference')) {
        if (this._destroyed) {
          return false;
        }
        const sr = normalizeStencilReference(reference);
        if (sr === null) {
          return false;
        }
        this._stencilReference = null;
        try {
          return this._bindStencilReferenceOnPass(refOrPass, sr) !== BIND_STATUS_UNSUPPORTED;
        } catch (err) {
          this._stats.passErrors += 1;
          this._warn('setStencilReference failed on pass', {
            error: String(err && err.message ? err.message : err),
          });
          return false;
        }
      }
      const sr = normalizeStencilReference(refOrPass);
      if (sr === null) {
        return false;
      }
      this._stencilReference = sr;
      return true;
    }

    _materializeStagedBindGroupDescriptor(slot) {
      const bit = 1 << slot;
      if ((this._stagedBgMask & bit) === 0) {
        return null;
      }
      const raw = this._stagedBgRaw[slot];
      const meta = this._stagedBgMeta[slot];
      let dynamicOffsets = null;
      if (this._stagedBgHasDyn[slot] === 1) {
        const src = this._stagedBgDynSource[slot];
        const start = this._stagedBgDynStart[slot];
        const explicitLen = this._stagedBgDynLength[slot];
        const len = explicitLen >= 0 ? explicitLen : src.length;
        if (start === 0 && len === src.length) {
          dynamicOffsets = src;
        } else {
          dynamicOffsets =
            typeof src.subarray === 'function'
              ? src.subarray(start, start + len)
              : Array.prototype.slice.call(src, start, start + len);
        }
      }
      return {
        groupIndex: slot,
        bindGroup: this._stagedBgGroups[slot],
        dynamicOffsets,
        entries:
          (meta && meta.entries) ||
          (raw && raw.entries) ||
          (raw && raw.desc && raw.desc.entries) ||
          null,
        hasCbv: Boolean((meta && meta.hasCbv) || (raw && raw.hasCbv)),
        hasTexture: Boolean((meta && meta.hasTexture) || (raw && raw.hasTexture)),
        hasSampler: Boolean((meta && meta.hasSampler) || (raw && raw.hasSampler)),
        cbvBuffer: (meta && meta.cbvBuffer) || (raw && raw.cbvBuffer) || null,
        textureView: (meta && meta.textureView) || (raw && raw.textureView) || null,
        sampler: (meta && meta.sampler) || (raw && raw.sampler) || null,
      };
    }

    /**
     * Return a shallow snapshot of the state currently tracked as bound on the active render pass.
     */
    getBoundPassState() {
      const bindGroups = {};
      let bgMask = this._boundBgMask;
      while (bgMask !== 0) {
        const bit = bgMask & -bgMask;
        const slot = 31 - Math.clz32(bit);
        bgMask ^= bit;
        let dynCopy = null;
        if (this._boundBgHasDyn[slot] === 1) {
          const len = this._boundBgDynLen[slot];
          dynCopy = new Array(len);
          if (len <= MAX_FAST_DYN_OFFSETS) {
            const base = slot * MAX_FAST_DYN_OFFSETS;
            for (let i = 0; i < len; i += 1) {
              dynCopy[i] = this._boundBgDynOffsets[base + i];
            }
          } else {
            const overflow = this._boundBgDynOverflow[slot];
            for (let i = 0; i < len; i += 1) {
              dynCopy[i] = overflow[i];
            }
          }
        }
        bindGroups[slot] = {
          bindGroup: this._boundBgGroups[slot],
          dynamicOffsets: dynCopy,
        };
      }
      if (this._boundBgOverflow !== null && this._boundBgOverflow.size > 0) {
        for (const [groupIndex, rec] of this._boundBgOverflow.entries()) {
          bindGroups[groupIndex] = {
            bindGroup: rec.bindGroup,
            dynamicOffsets: rec.hasDynamicOffsets
              ? rec.dynamicOffsetsSnapshot
                ? rec.dynamicOffsetsSnapshot.slice(0, rec.dynamicOffsetsLength)
                : []
              : null,
          };
        }
      }

      const vertexBuffers = {};
      let vbMask = this._boundVbMask;
      while (vbMask !== 0) {
        const bit = vbMask & -vbMask;
        const slot = 31 - Math.clz32(bit);
        vbMask ^= bit;
        vertexBuffers[slot] = {
          buffer: this._boundVbBuffers[slot],
          offset: this._boundVbOffsets[slot],
          size: this._boundVbSizes[slot],
        };
      }
      if (this._boundVbOverflow !== null && this._boundVbOverflow.size > 0) {
        for (const [slot, rec] of this._boundVbOverflow.entries()) {
          vertexBuffers[slot] = {
            buffer: rec.buffer,
            offset: rec.offset,
            size: rec.size,
          };
        }
      }

      return {
        hasActivePass: this._boundPass !== null,
        pipeline: this._boundPipeline,
        bindGroups,
        vertexBuffers,
        indexBuffer: this._hasBoundIb
          ? {
              buffer: this._boundIbBuffer,
              indexFormat: this._boundIbFormat,
              offset: this._boundIbOffset,
              size: this._boundIbSize,
            }
          : null,
        viewport: this._hasBoundViewport ? { ...this._boundViewportData } : null,
        scissorRect: this._hasBoundScissor ? { ...this._boundScissorData } : null,
        blendConstant: this._hasBoundBlendConstant ? { ...this._boundBlendConstantData } : null,
        stencilReference: this._boundStencilReference,
      };
    }

    /**
     * Snapshot active draw state merged with optional per-draw overrides.
     */
    getDrawState(overrides) {
      const ov = overrides && typeof overrides === 'object' ? overrides : {};
      const pipe = ov.pipeline !== undefined ? ov.pipeline : this._activePipeline;

      const vertexBuffers = [];
      let vbMask = this._stagedVbMask;
      while (vbMask !== 0) {
        const bit = vbMask & -vbMask;
        const slot = 31 - Math.clz32(bit);
        vbMask ^= bit;
        vertexBuffers.push({
          slot,
          buffer: this._stagedVbBuffers[slot],
          offset: this._stagedVbOffsets[slot],
          size: this._stagedVbSizes[slot],
          stride: this._stagedVbStrides[slot],
        });
      }
      if (this._stagedVbOverflow !== null && this._stagedVbOverflow.size > 0) {
        for (const vb of this._stagedVbOverflow.values()) {
          vertexBuffers.push({ ...vb });
        }
      }
      if (Array.isArray(ov.vertexBuffers)) {
        for (let i = 0; i < ov.vertexBuffers.length; i += 1) {
          if (ov.vertexBuffers[i]) {
            vertexBuffers.push(ov.vertexBuffers[i]);
          }
        }
      }

      const bindGroups = [];
      let bgMask = this._stagedBgMask;
      while (bgMask !== 0) {
        const bit = bgMask & -bgMask;
        const slot = 31 - Math.clz32(bit);
        bgMask ^= bit;
        bindGroups.push(this._materializeStagedBindGroupDescriptor(slot));
      }
      if (this._stagedBgOverflow !== null && this._stagedBgOverflow.size > 0) {
        for (const bg of this._stagedBgOverflow.values()) {
          bindGroups.push(bg);
        }
      }
      if (Array.isArray(ov.bindGroups)) {
        for (let i = 0; i < ov.bindGroups.length; i += 1) {
          if (ov.bindGroups[i]) {
            bindGroups.push(ov.bindGroups[i]);
          }
        }
      }

      const cbvs = Array.from(this._cbvs.values());
      const textures = Array.from(this._textures.values());
      const samplers = Array.from(this._samplers.values());

      let vertexStride =
        ov.vertexStride !== undefined
          ? ov.vertexStride
          : pipe && typeof pipe === 'object' && typeof pipe.vertexStride === 'number'
            ? pipe.vertexStride
            : null;
      if (vertexStride === null && (this._stagedVbMask & 1) !== 0) {
        if (typeof this._stagedVbStrides[0] === 'number') {
          vertexStride = this._stagedVbStrides[0];
        }
      }

      const state = {
        vertexCount: ov.vertexCount !== undefined ? ov.vertexCount : 0,
        indexCount: ov.indexCount !== undefined ? ov.indexCount : 0,
        instanceCount: ov.instanceCount !== undefined ? ov.instanceCount : 1,
        firstVertex: ov.firstVertex !== undefined ? ov.firstVertex : 0,
        firstIndex: ov.firstIndex !== undefined ? ov.firstIndex : 0,
        baseVertex: ov.baseVertex !== undefined ? ov.baseVertex : 0,
        firstInstance: ov.firstInstance !== undefined ? ov.firstInstance : 0,
        indexed: Boolean(ov.indexed),
        pipeline: pipe,
        vertexStride,
        vertexBuffers,
        vertexBufferLayouts:
          ov.vertexBufferLayouts ||
          (pipe && typeof pipe === 'object' ? pipe.vertexBufferLayouts || pipe.vertexBuffers : null) ||
          null,
        bindGroups,
        cbvs,
        textures,
        samplers,
        viewport:
          ov.viewport !== undefined
            ? ov.viewport
            : this._hasStagedViewport
              ? { ...this._stagedViewport }
              : null,
        scissorRect:
          ov.scissorRect !== undefined
            ? ov.scissorRect
            : ov.scissor !== undefined
              ? ov.scissor
              : this._hasStagedScissor
                ? { ...this._stagedScissor }
                : null,
        blendConstant:
          ov.blendConstant !== undefined
            ? ov.blendConstant
            : this._hasStagedBlendConstant
              ? { ...this._stagedBlendConstant }
              : null,
        stencilReference:
          ov.stencilReference !== undefined ? ov.stencilReference : this._stencilReference,
      };

      // Propagate explicit override hints when passed directly to executeDraw / resolve.
      for (let i = 0; i < PASSTHROUGH_DRAW_STATE_KEYS.length; i += 1) {
        const key = PASSTHROUGH_DRAW_STATE_KEYS[i];
        if (ov[key] !== undefined) {
          state[key] = ov[key];
        }
      }

      return state;
    }

    _needsFallbackRouting(drawState) {
      if (drawState.isFallback === true || drawState.usesFallbackShaders === true) {
        return true;
      }
      const pipe = drawState.pipeline;
      if (!pipe) {
        return true;
      }
      if (typeof pipe === 'object') {
        if (pipe.isFallback === true || pipe.usesFallbackShaders === true || pipe.missingShaders === true) {
          return true;
        }
        const vs = drawState.vertexShader || pipe.vertexShader;
        const fs = drawState.fragmentShader || pipe.fragmentShader;
        if (
          (vs && (vs.missing === true || vs.isFallback === true || vs.source === 'builtin' || vs.source === 'fallback')) ||
          (fs && (fs.missing === true || fs.isFallback === true || fs.source === 'builtin' || fs.source === 'fallback'))
        ) {
          return true;
        }
        // Descriptor object with no compiled GPURenderPipeline handle attached.
        if (!pipe.pipeline && typeof pipe.getBindGroupLayout !== 'function') {
          return true;
        }
      }
      return false;
    }

    _needsFallbackRoutingFast(args, pipe) {
      if (args !== null && (args.isFallback === true || args.usesFallbackShaders === true)) {
        return true;
      }
      if (!pipe) {
        return true;
      }
      if (typeof pipe === 'object') {
        if (pipe.isFallback === true || pipe.usesFallbackShaders === true || pipe.missingShaders === true) {
          return true;
        }
        const vs = (args !== null && args.vertexShader) || pipe.vertexShader;
        const fs = (args !== null && args.fragmentShader) || pipe.fragmentShader;
        if (
          (vs && (vs.missing === true || vs.isFallback === true || vs.source === 'builtin' || vs.source === 'fallback')) ||
          (fs && (fs.missing === true || fs.isFallback === true || fs.source === 'builtin' || fs.source === 'fallback'))
        ) {
          return true;
        }
        if (!pipe.pipeline && typeof pipe.getBindGroupLayout !== 'function') {
          return true;
        }
      }
      return false;
    }

    _extractShaderHash(drawState, stageName) {
      const pipe = drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
      const stageObj =
        stageName === 'vertex'
          ? drawState.vertexShader || (pipe && pipe.vertexShader)
          : drawState.fragmentShader || (pipe && pipe.fragmentShader);
      const explicitHash =
        stageName === 'vertex'
          ? drawState.vertexShaderHash || (pipe && pipe.vertexShaderHash)
          : drawState.fragmentShaderHash || (pipe && pipe.fragmentShaderHash);

      if (explicitHash !== undefined && explicitHash !== null && explicitHash !== '') {
        return explicitHash;
      }
      if (stageObj && typeof stageObj === 'object') {
        if (stageObj.requestedHash !== undefined && stageObj.requestedHash !== null) {
          return stageObj.requestedHash;
        }
        if (stageObj.hash !== undefined && stageObj.hash !== null) {
          return stageObj.hash;
        }
        if (stageObj.hashHex !== undefined && stageObj.hashHex !== null) {
          return stageObj.hashHex;
        }
      }
      const pipeId = pipe && pipe.id !== undefined ? pipe.id : 0;
      return fnv1a64Hex(`missing:${stageName}:${pipeId}`);
    }

    _invokeRouterResolve(hash, stageName, profile) {
      const router = this._shaderFallbackRouter;
      if (!router || typeof router.resolve !== 'function') {
        return {
          wgsl: EMERGENCY_BUILTIN_WGSL[stageName],
          source: 'builtin',
          entryPoint: stageName === 'fragment' ? 'fs_main' : 'vs_main',
          resolvedHash: null,
          profile: FALLBACK_PROFILE.DEFAULT,
        };
      }

      let raw = null;
      try {
        raw = router.resolve(hash, stageName, { profile });
      } catch (_err3) {
        // Backwards compatibility with strict legacy 2-arg mock routers that reject a 3rd options arg.
        this._stats.legacyRouterFallbacks += 1;
        try {
          raw = router.resolve(hash, stageName);
        } catch (err2) {
          this._warn('ShaderFallbackRouter.resolve failed', {
            stage: stageName,
            profile,
            error: String(err2 && err2.message ? err2.message : err2),
          });
          return null;
        }
      }

      return unwrapRouterResult(raw, stageName);
    }

    /**
     * Select and resolve fallback vertex + fragment WGSL for a draw state.
     *
     * @param {object} drawState
     * @param {string} [overrideProfile]
     */
    resolveFallbackShaders(drawState, overrideProfile) {
      const state = drawState && typeof drawState === 'object' ? drawState : this.getDrawState();
      const selectedProfile =
        overrideProfile !== undefined && overrideProfile !== null
          ? normalizeProfileName(overrideProfile)
          : selectFallbackShaderProfile(state);

      const vsHash = this._extractShaderHash(state, 'vertex');
      const fsHash = this._extractShaderHash(state, 'fragment');

      const vertex = this._invokeRouterResolve(vsHash, 'vertex', selectedProfile);
      const fragment = this._invokeRouterResolve(fsHash, 'fragment', selectedProfile);

      if (!vertex || !fragment) {
        return null;
      }

      const effectiveProfile = inferEffectiveProfileFromWgsl(vertex, fragment);
      return {
        selectedProfile,
        effectiveProfile,
        vertex,
        fragment,
      };
    }

    _getOrCreateShaderModule(wgsl, label) {
      if (!this._device || typeof this._device.createShaderModule !== 'function') {
        return null;
      }
      const key = this._hashWgsl(wgsl);
      const cached = this._fallbackShaderModuleCache.get(key);
      if (cached) {
        return cached;
      }
      const module = this._device.createShaderModule({ code: wgsl, label });
      if (this._fallbackShaderModuleCache.size >= this._maxFallbackPipelines * 2) {
        const oldestKey = this._fallbackShaderModuleCache.keys().next().value;
        this._fallbackShaderModuleCache.delete(oldestKey);
      }
      this._fallbackShaderModuleCache.set(key, module);
      return module;
    }

    _getOrCreateFallbackPipeline(resolved, drawState) {
      if (!this._device || typeof this._device.createRenderPipeline !== 'function') {
        return null;
      }

      const pipe = drawState.pipeline && typeof drawState.pipeline === 'object' ? drawState.pipeline : null;
      const format = drawState.format || (pipe && pipe.format) || this._format || DEFAULT_COLOR_FORMAT;
      const rawTopology = drawState.topology || (pipe && pipe.topology) || DEFAULT_TOPOLOGY;
      const topology = isRectListTopology(rawTopology) ? DEFAULT_TOPOLOGY : rawTopology;
      const sampleCount = toNonNegativeInt(
        drawState.sampleCount !== undefined
          ? drawState.sampleCount
          : pipe && pipe.sampleCount !== undefined
            ? pipe.sampleCount
            : 1,
        1
      ) || 1;
      const depthStencilFormat =
        drawState.depthStencilFormat !== undefined
          ? drawState.depthStencilFormat
          : pipe && pipe.depthStencilFormat !== undefined
            ? pipe.depthStencilFormat
            : this._depthStencilFormat;

      const vsKey = this._hashWgsl(resolved.vertex.wgsl);
      const fsKey = this._hashWgsl(resolved.fragment.wgsl);
      const cacheKey = `${resolved.effectiveProfile}|${vsKey}|${fsKey}|${format}|${topology}|${sampleCount}|${depthStencilFormat || 'none'}`;

      const cached = this._fallbackPipelineCache.get(cacheKey);
      if (cached) {
        this._stats.fallbackPipelineCacheHits += 1;
        return cached;
      }

      this._stats.fallbackPipelineCacheMisses += 1;
      try {
        const vsModule = this._getOrCreateShaderModule(
          resolved.vertex.wgsl,
          `fallback:${resolved.effectiveProfile}:vs`
        );
        const fsModule = this._getOrCreateShaderModule(
          resolved.fragment.wgsl,
          `fallback:${resolved.effectiveProfile}:fs`
        );
        if (!vsModule || !fsModule) {
          this._stats.pipelineErrors += 1;
          return null;
        }

        const vertexBuffers =
          resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B
            ? [
                {
                  arrayStride: FRAME1_SPLASH_VERTEX_LAYOUT_24B.arrayStride,
                  stepMode: FRAME1_SPLASH_VERTEX_LAYOUT_24B.stepMode,
                  attributes: FRAME1_SPLASH_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
                },
              ]
            : resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B
              ? [
                  {
                    arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
                    stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
                    attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
                  },
                ]
              : [];

        const descriptor = {
          label: `draw_executor:fallback:${resolved.effectiveProfile}`,
          layout: 'auto',
          vertex: {
            module: vsModule,
            entryPoint: resolved.vertex.entryPoint || 'vs_main',
            buffers: vertexBuffers,
          },
          fragment: {
            module: fsModule,
            entryPoint: resolved.fragment.entryPoint || 'fs_main',
            targets: [{ format }],
          },
          primitive: {
            topology,
          },
        };

        if (sampleCount > 1) {
          descriptor.multisample = { count: sampleCount };
        }
        if (depthStencilFormat) {
          descriptor.depthStencil = {
            format: depthStencilFormat,
            depthWriteEnabled: false,
            depthCompare: 'always',
          };
        }

        const pipeline =
          this._pipelineCache && typeof this._pipelineCache.getOrCreateRenderPipeline === 'function'
            ? this._pipelineCache.getOrCreateRenderPipeline(descriptor)
            : this._device.createRenderPipeline(descriptor);

        if (!pipeline) {
          this._stats.pipelineErrors += 1;
          return null;
        }

        if (this._fallbackPipelineCache.size >= this._maxFallbackPipelines) {
          const oldestKey = this._fallbackPipelineCache.keys().next().value;
          this._fallbackPipelineCache.delete(oldestKey);
        }
        this._fallbackPipelineCache.set(cacheKey, pipeline);
        return pipeline;
      } catch (err) {
        this._stats.pipelineErrors += 1;
        this._warn('Failed to create fallback render pipeline', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackCbvBuffer() {
      if (this._fallbackCbvBuffer && this._fallbackCbvBuffer.destroyed !== true) {
        return this._fallbackCbvBuffer;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_cbv_256b',
          size: FRAME1_CBV_MIN_BYTES,
          usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          // Deterministic non-zero seed bytes (all finite float values) so Frame-1 modulation is visible.
          const seed = new Uint32Array(FRAME1_CBV_MIN_BYTES / 4);
          seed[0] = 0x3f204080;
          seed[1] = 0x3f4080c0;
          seed[2] = 0x3f60a0e0;
          seed[3] = 0x3f800000;
          this._device.queue.writeBuffer(buf, 0, seed);
        }
        this._fallbackCbvBuffer = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic fallback CBV buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackTextureView() {
      if (
        this._fallbackTexture &&
        this._fallbackTexture.destroyed !== true &&
        this._fallbackTextureView
      ) {
        return this._fallbackTextureView;
      }
      if (!this._device || typeof this._device.createTexture !== 'function') {
        return null;
      }
      try {
        const tex = this._device.createTexture({
          label: 'draw_executor:fallback_tex_1x1',
          size: { width: 1, height: 1, depthOrArrayLayers: 1 },
          mipLevelCount: 1,
          format: 'rgba8unorm',
          usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
          dimension: '2d',
        });
        if (!tex) {
          return null;
        }
        if (this._device.queue && typeof this._device.queue.writeTexture === 'function') {
          const pixel = new Uint8Array([0x28, 0x38, 0x50, 0xff]);
          this._device.queue.writeTexture(
            { texture: tex },
            pixel,
            { bytesPerRow: 4, rowsPerImage: 1 },
            { width: 1, height: 1, depthOrArrayLayers: 1 }
          );
        }
        const view = typeof tex.createView === 'function' ? tex.createView() : tex;
        this._fallbackTexture = tex;
        this._fallbackTextureView = view;
        return view;
      } catch (err) {
        this._warn('Failed to create synthetic fallback texture', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackSampler() {
      if (this._fallbackSampler && this._fallbackSampler.destroyed !== true) {
        return this._fallbackSampler;
      }
      if (!this._device || typeof this._device.createSampler !== 'function') {
        return null;
      }
      try {
        this._fallbackSampler = this._device.createSampler({
          label: 'draw_executor:fallback_sampler',
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        });
        return this._fallbackSampler;
      } catch (err) {
        this._warn('Failed to create synthetic fallback sampler', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackVertexBuffer24B() {
      if (this._fallbackVertexBuffer24B && this._fallbackVertexBuffer24B.destroyed !== true) {
        return this._fallbackVertexBuffer24B;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        // 6 vertices * 24 bytes (vec4<f32> pos + vec2<f32> uv) = 144 bytes (fullscreen quad).
        const quad = new Float32Array([
          // Triangle 1
          -1.0, -1.0, 0.0, 1.0, 0.0, 1.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
          -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
          // Triangle 2
          -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 0.0,
        ]);
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_vb_24b_quad',
          size: quad.byteLength,
          usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          this._device.queue.writeBuffer(buf, 0, quad);
        }
        this._fallbackVertexBuffer24B = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic 24B fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _ensureFallbackSplashVertexBuffer24B() {
      if (
        this._fallbackSplashVertexBuffer24B &&
        this._fallbackSplashVertexBuffer24B.destroyed !== true
      ) {
        return this._fallbackSplashVertexBuffer24B;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return null;
      }
      try {
        // 3 RECTLIST vertices in exact Frame-1 24B layout (float32x3 pos, unorm8x4 color, float32x2 uv)
        // expanded to 6 triangle-list vertices (144 bytes).
        const rect3 = new Uint8Array(3 * TEXTURED_QUAD_VERTEX_STRIDE);
        const view = new DataView(rect3.buffer);
        const corners = [
          { x: -1.0, y: 1.0, z: 0.0, u: 0.0, v: 0.0 },
          { x: 1.0, y: 1.0, z: 0.0, u: 1.0, v: 0.0 },
          { x: -1.0, y: -1.0, z: 0.0, u: 0.0, v: 1.0 },
        ];
        for (let i = 0; i < 3; i += 1) {
          const base = i * TEXTURED_QUAD_VERTEX_STRIDE;
          view.setFloat32(base + 0, corners[i].x, true);
          view.setFloat32(base + 4, corners[i].y, true);
          view.setFloat32(base + 8, corners[i].z, true);
          rect3[base + 12] = 0xff;
          rect3[base + 13] = 0xff;
          rect3[base + 14] = 0xff;
          rect3[base + 15] = 0xff;
          view.setFloat32(base + 16, corners[i].u, true);
          view.setFloat32(base + 20, corners[i].v, true);
        }
        const expanded = expandRectListVertices24B(rect3, 3);
        const buf = this._device.createBuffer({
          label: 'draw_executor:fallback_vb_24b_splash_rect',
          size: expanded.byteLength,
          usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
        });
        if (buf && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          this._device.queue.writeBuffer(buf, 0, expanded);
        }
        this._fallbackSplashVertexBuffer24B = buf;
        return buf;
      } catch (err) {
        this._warn('Failed to create synthetic 24B splash rect fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return null;
      }
    }

    _getStagedBg0Entries() {
      if ((this._stagedBgMask & 1) === 0) {
        return null;
      }
      const meta = this._stagedBgMeta[0];
      if (meta && Array.isArray(meta.entries)) {
        return meta.entries;
      }
      const raw = this._stagedBgRaw[0];
      if (raw) {
        if (Array.isArray(raw.entries)) {
          return raw.entries;
        }
        if (raw.desc && Array.isArray(raw.desc.entries)) {
          return raw.desc.entries;
        }
      }
      return null;
    }

    _findBoundCbvCandidate(drawState) {
      if (drawState.cbvBuffer) {
        return { buffer: drawState.cbvBuffer, offset: 0, size: null };
      }
      if (drawState.uniformBuffer) {
        return { buffer: drawState.uniformBuffer, offset: 0, size: null };
      }
      if (drawState.cbv && typeof drawState.cbv === 'object') {
        return drawState.cbv.buffer !== undefined
          ? drawState.cbv
          : { buffer: drawState.cbv, offset: 0, size: null };
      }
      if (this._cbvs.has(0)) {
        return this._cbvs.get(0);
      }
      for (const entry of this._cbvs.values()) {
        return entry;
      }
      if ((this._stagedBgMask & 1) !== 0) {
        const meta = this._stagedBgMeta[0];
        const raw = this._stagedBgRaw[0];
        const cbvBuffer = (meta && meta.cbvBuffer) || (raw && raw.cbvBuffer) || null;
        if (cbvBuffer) {
          return { buffer: cbvBuffer, offset: 0, size: null };
        }
        const entries = this._getStagedBg0Entries();
        if (entries) {
          for (let i = 0; i < entries.length; i += 1) {
            const e = entries[i];
            if (e && e.resource && typeof e.resource === 'object' && e.resource.buffer) {
              return {
                buffer: e.resource.buffer,
                offset: e.resource.offset || 0,
                size: e.resource.size !== undefined ? e.resource.size : null,
              };
            }
          }
        }
      }
      return null;
    }

    _findBoundTextureCandidate(drawState) {
      if (drawState.textureView) return drawState.textureView;
      if (drawState.texture) return drawState.texture;
      if (drawState.srv) return drawState.srv;
      if (this._textures.has(0)) return this._textures.get(0);
      for (const tex of this._textures.values()) {
        return tex;
      }
      if ((this._stagedBgMask & 1) !== 0) {
        const meta = this._stagedBgMeta[0];
        const raw = this._stagedBgRaw[0];
        const textureView = (meta && meta.textureView) || (raw && raw.textureView) || null;
        if (textureView) return textureView;
        const entries = this._getStagedBg0Entries();
        if (entries) {
          for (let i = 0; i < entries.length; i += 1) {
            const e = entries[i];
            if (!e) continue;
            if (e.textureView) return e.textureView;
            if (e.texture) return e.texture;
            if (
              e.resource &&
              typeof e.resource === 'object' &&
              (e.type === 'texture' ||
                e.resourceType === 1 ||
                e.resourceType === 'textureView' ||
                e.resource.kind === 'GPUTextureView' ||
                e.resource.kind === 'GPUTexture')
            ) {
              return e.resource;
            }
          }
        }
      }
      return null;
    }

    _findBoundSamplerCandidate(drawState) {
      if (drawState.sampler) return drawState.sampler;
      if (this._samplers.has(0)) return this._samplers.get(0);
      for (const s of this._samplers.values()) {
        return s;
      }
      if ((this._stagedBgMask & 1) !== 0) {
        const meta = this._stagedBgMeta[0];
        const raw = this._stagedBgRaw[0];
        const sampler = (meta && meta.sampler) || (raw && raw.sampler) || null;
        if (sampler) return sampler;
        const entries = this._getStagedBg0Entries();
        if (entries) {
          for (let i = 0; i < entries.length; i += 1) {
            const e = entries[i];
            if (!e) continue;
            if (e.sampler) return e.sampler;
            if (
              e.resource &&
              typeof e.resource === 'object' &&
              (e.type === 'sampler' ||
                e.resourceType === 2 ||
                e.resourceType === 'sampler' ||
                e.resource.kind === 'GPUSampler')
            ) {
              return e.resource;
            }
          }
        }
      }
      return null;
    }

    _bindFallbackGroupForCbv(pass, pipeline, drawState) {
      const out = this._tmpFallbackBgResult;
      out.ok = false;
      out.substitutedCbv = false;
      out.substitutedTexture = false;
      out.substitutedSampler = false;
      out.cacheHit = false;
      out.bindGroupElided = false;

      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return out;
      }

      const candidate = this._findBoundCbvCandidate(drawState);
      let buffer;
      let offset = 0;
      let size = FRAME1_CBV_MIN_BYTES;
      let substitutedCbv = false;

      if (candidate && isValidCbvBinding(candidate)) {
        buffer = candidate.buffer !== undefined ? candidate.buffer : candidate;
        offset = candidate.offset || 0;
        if (candidate.size !== undefined && candidate.size !== null) {
          size = candidate.size;
        } else if (typeof buffer.size === 'number' && Number.isFinite(buffer.size)) {
          size = Math.max(FRAME1_CBV_MIN_BYTES, buffer.size - offset);
        }
      } else {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidCbvSubstitutions += 1;
        substitutedCbv = true;
        buffer = this._ensureFallbackCbvBuffer();
        if (!buffer) {
          this._stats.bindGroupErrors += 1;
          out.substitutedCbv = true;
          return out;
        }
      }

      try {
        const bindings = this._bgScratchBindings;
        const kinds = this._bgScratchKinds;
        const resA = this._bgScratchResA;
        const resB = this._bgScratchResB;
        const genA = this._bgScratchGenA;
        const genB = this._bgScratchGenB;
        const offs = this._bgScratchOff;
        const sizes = this._bgScratchSize;
        const bufGen = getResourceGeneration(buffer);
        bindings[0] = 0;
        kinds[0] = BG_ENTRY_KIND_BUFFER;
        resA[0] = buffer;
        resB[0] = null;
        genA[0] = bufGen;
        genB[0] = 0;
        offs[0] = offset;
        sizes[0] = size;

        let h = -2128831035 | 0;
        h = hashMixInt32(h, BG_OWNER_FALLBACK_CBV);
        h = hashMixInt32(h, this._getResourceId(pipeline));
        h = hashMixInt32(h, this._getResourceId(buffer));
        h = hashMixInt32(h, bufGen);
        h = hashMixNum(h, offset);
        h = hashMixNum(h, size);

        let bindGroup = this._bgLookupNode(
          h,
          BG_OWNER_FALLBACK_CBV,
          pipeline,
          0,
          1,
          bindings,
          kinds,
          resA,
          resB,
          genA,
          genB,
          offs,
          sizes
        );
        let cacheHit = true;
        if (!bindGroup) {
          cacheHit = false;
          this._stats.bindGroupCacheMisses += 1;
          const layout = this._getPipelineBindGroupLayout(pipeline, 0);
          bindGroup = this._device.createBindGroup({
            label: 'draw_executor:fallback_cbv_bg',
            layout,
            entries: [
              {
                binding: 0,
                resource: { buffer, offset, size },
              },
            ],
          });
          this._bgStoreNode(
            h,
            BG_OWNER_FALLBACK_CBV,
            pipeline,
            0,
            1,
            bindings,
            kinds,
            resA,
            resB,
            genA,
            genB,
            offs,
            sizes,
            bindGroup,
            [buffer]
          );
        }
        resA[0] = null;
        const bound = this._bindBindGroupOnPass(pass, 0, bindGroup, null);
        out.ok = true;
        out.substitutedCbv = substitutedCbv;
        out.cacheHit = cacheHit;
        out.bindGroupElided = !bound;
        return out;
      } catch (err) {
        this._bgScratchResA[0] = null;
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback CBV bind group', {
          error: String(err && err.message ? err.message : err),
        });
        out.substitutedCbv = substitutedCbv;
        return out;
      }
    }

    _bindFallbackGroupForTexturedQuad(pass, pipeline, drawState) {
      const out = this._tmpFallbackBgResult;
      out.ok = false;
      out.substitutedCbv = false;
      out.substitutedTexture = false;
      out.substitutedSampler = false;
      out.cacheHit = false;
      out.bindGroupElided = false;

      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return out;
      }

      let substitutedTexture = false;
      let substitutedSampler = false;

      const rawTexture = this._findBoundTextureCandidate(drawState);
      let textureView = this._resolveTextureViewCached(rawTexture);
      let sourceTexture = rawTexture && rawTexture.texture ? rawTexture.texture : rawTexture;
      if (!textureView) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidTextureSubstitutions += 1;
        substitutedTexture = true;
        textureView = this._ensureFallbackTextureView();
        sourceTexture = this._fallbackTexture;
        if (!textureView) {
          this._stats.bindGroupErrors += 1;
          out.substitutedTexture = true;
          return out;
        }
      }

      let sampler = unwrapSampler(this._findBoundSamplerCandidate(drawState));
      if (!sampler) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidSamplerSubstitutions += 1;
        substitutedSampler = true;
        sampler = this._ensureFallbackSampler();
        if (!sampler) {
          this._stats.bindGroupErrors += 1;
          out.substitutedTexture = substitutedTexture;
          out.substitutedSampler = true;
          return out;
        }
      }

      try {
        const bindings = this._bgScratchBindings;
        const kinds = this._bgScratchKinds;
        const resA = this._bgScratchResA;
        const resB = this._bgScratchResB;
        const genA = this._bgScratchGenA;
        const genB = this._bgScratchGenB;
        const offs = this._bgScratchOff;
        const sizes = this._bgScratchSize;
        const viewGen = getResourceGeneration(textureView);
        const smpGen = getResourceGeneration(sampler);
        const texSrc = sourceTexture && sourceTexture !== textureView ? sourceTexture : null;
        const srcGen = texSrc ? getResourceGeneration(texSrc) : 0;

        bindings[0] = 0;
        kinds[0] = BG_ENTRY_KIND_TEXTURE;
        resA[0] = texSrc;
        resB[0] = textureView;
        genA[0] = srcGen;
        genB[0] = viewGen;
        offs[0] = 0;
        sizes[0] = -1;

        bindings[1] = 1;
        kinds[1] = BG_ENTRY_KIND_RAW;
        resA[1] = sampler;
        resB[1] = null;
        genA[1] = smpGen;
        genB[1] = 0;
        offs[1] = 0;
        sizes[1] = -1;

        let h = -2128831035 | 0;
        h = hashMixInt32(h, BG_OWNER_FALLBACK_TEX);
        h = hashMixInt32(h, this._getResourceId(pipeline));
        h = hashMixInt32(h, this._getResourceId(textureView));
        h = hashMixInt32(h, viewGen);
        h = hashMixInt32(h, this._getResourceId(sampler));
        h = hashMixInt32(h, smpGen);

        let bindGroup = this._bgLookupNode(
          h,
          BG_OWNER_FALLBACK_TEX,
          pipeline,
          0,
          2,
          bindings,
          kinds,
          resA,
          resB,
          genA,
          genB,
          offs,
          sizes
        );
        let cacheHit = true;
        if (!bindGroup) {
          cacheHit = false;
          this._stats.bindGroupCacheMisses += 1;
          const layout = this._getPipelineBindGroupLayout(pipeline, 0);
          bindGroup = this._device.createBindGroup({
            label: 'draw_executor:fallback_textured_quad_bg',
            layout,
            entries: [
              { binding: 0, resource: textureView },
              { binding: 1, resource: sampler },
            ],
          });
          const tracked = texSrc
            ? [texSrc, textureView, sampler]
            : [textureView, sampler];
          this._bgStoreNode(
            h,
            BG_OWNER_FALLBACK_TEX,
            pipeline,
            0,
            2,
            bindings,
            kinds,
            resA,
            resB,
            genA,
            genB,
            offs,
            sizes,
            bindGroup,
            tracked
          );
        }
        resA[0] = null;
        resB[0] = null;
        resA[1] = null;
        const bound = this._bindBindGroupOnPass(pass, 0, bindGroup, null);
        out.ok = true;
        out.substitutedTexture = substitutedTexture;
        out.substitutedSampler = substitutedSampler;
        out.cacheHit = cacheHit;
        out.bindGroupElided = !bound;
        return out;
      } catch (err) {
        this._bgScratchResA[0] = null;
        this._bgScratchResB[0] = null;
        this._bgScratchResA[1] = null;
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback textured quad bind group', {
          error: String(err && err.message ? err.message : err),
        });
        out.substitutedTexture = substitutedTexture;
        out.substitutedSampler = substitutedSampler;
        return out;
      }
    }

    _bindFallbackVertexBuffer24B(pass, drawState, vertexCount, firstVertex) {
      const out = this._tmpFallbackVbResult;
      out.ok = false;
      out.substitutedVertexBuffer = false;
      out.vertexBufferElided = false;
      out.effectiveVertexCount = 0;
      out.effectiveFirstVertex = 0;

      const hasSlot0 = (this._stagedVbMask & 1) !== 0;
      const fallbackSlot0 = !hasSlot0 && Array.isArray(drawState.vertexBuffers)
        ? drawState.vertexBuffers[0]
        : null;
      const requiredBytes = drawState.indexed
        ? TEXTURED_QUAD_VERTEX_STRIDE
        : (firstVertex + vertexCount) * TEXTURED_QUAD_VERTEX_STRIDE;

      const slot0Valid = hasSlot0
        ? isValidVertexBufferParts(
            this._stagedVbBuffers[0],
            this._stagedVbOffsets[0],
            this._stagedVbSizes[0],
            this._stagedVbStrides[0],
            requiredBytes,
            TEXTURED_QUAD_VERTEX_STRIDE
          )
        : fallbackSlot0
          ? isValidVertexBufferBinding(fallbackSlot0, requiredBytes, TEXTURED_QUAD_VERTEX_STRIDE)
          : false;

      if (slot0Valid) {
        const buffer = hasSlot0
          ? this._stagedVbBuffers[0]
          : fallbackSlot0.buffer !== undefined
            ? fallbackSlot0.buffer
            : fallbackSlot0;
        const offset = hasSlot0 ? this._stagedVbOffsets[0] || 0 : fallbackSlot0.offset || 0;
        const size = hasSlot0 ? this._stagedVbSizes[0] : fallbackSlot0.size;
        try {
          const bound = this._bindVertexBufferOnPass(pass, 0, buffer, offset, size);
          out.ok = true;
          out.vertexBufferElided = !bound;
          out.effectiveVertexCount = vertexCount;
          out.effectiveFirstVertex = firstVertex;
          return out;
        } catch (err) {
          this._warn('setVertexBuffer threw on bound 24B vertex buffer; substituting synthetic quad', {
            error: String(err && err.message ? err.message : err),
          });
        }
      }

      this._stats.invalidVertexBufferGuards += 1;
      this._stats.vertexBufferSubstitutions += 1;
      out.substitutedVertexBuffer = true;
      const fallbackVb = this._ensureFallbackVertexBuffer24B();
      if (!fallbackVb) {
        return out;
      }

      try {
        const bound = this._bindVertexBufferOnPass(pass, 0, fallbackVb, 0, null);
        // Synthetic fallback quad holds 6 vertices (2 triangles); clamp non-indexed draw range safely.
        const clampedCount = drawState.indexed ? vertexCount : Math.min(Math.max(vertexCount, 3), 6);
        out.ok = true;
        out.vertexBufferElided = !bound;
        out.effectiveVertexCount = clampedCount;
        out.effectiveFirstVertex = 0;
        return out;
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('Failed to bind synthetic 24B fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return out;
      }
    }

    _bindFallbackGroupForSplashRect(pass, pipeline, drawState) {
      const out = this._tmpFallbackBgResult;
      out.ok = false;
      out.substitutedCbv = false;
      out.substitutedTexture = false;
      out.substitutedSampler = false;
      out.cacheHit = false;
      out.bindGroupElided = false;

      if (
        !this._device ||
        typeof this._device.createBindGroup !== 'function' ||
        typeof pipeline.getBindGroupLayout !== 'function'
      ) {
        this._stats.bindGroupErrors += 1;
        return out;
      }

      const cbvCandidate = this._findBoundCbvCandidate(drawState);
      let cbvBuffer;
      let cbvOffset = 0;
      let cbvSize = FRAME1_CBV_MIN_BYTES;
      let substitutedCbv = false;
      let substitutedTexture = false;
      let substitutedSampler = false;

      if (cbvCandidate && isValidCbvBinding(cbvCandidate)) {
        cbvBuffer = cbvCandidate.buffer !== undefined ? cbvCandidate.buffer : cbvCandidate;
        cbvOffset = cbvCandidate.offset || 0;
        if (cbvCandidate.size !== undefined && cbvCandidate.size !== null) {
          cbvSize = cbvCandidate.size;
        } else if (typeof cbvBuffer.size === 'number' && Number.isFinite(cbvBuffer.size)) {
          cbvSize = Math.max(FRAME1_CBV_MIN_BYTES, cbvBuffer.size - cbvOffset);
        }
      } else {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidCbvSubstitutions += 1;
        substitutedCbv = true;
        cbvBuffer = this._ensureFallbackCbvBuffer();
        if (!cbvBuffer) {
          this._stats.bindGroupErrors += 1;
          out.substitutedCbv = true;
          return out;
        }
      }

      const rawTexture = this._findBoundTextureCandidate(drawState);
      let textureView = this._resolveTextureViewCached(rawTexture);
      let sourceTexture = rawTexture && rawTexture.texture ? rawTexture.texture : rawTexture;
      if (!textureView) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidTextureSubstitutions += 1;
        substitutedTexture = true;
        textureView = this._ensureFallbackTextureView();
        sourceTexture = this._fallbackTexture;
        if (!textureView) {
          this._stats.bindGroupErrors += 1;
          out.substitutedCbv = substitutedCbv;
          out.substitutedTexture = true;
          return out;
        }
      }

      let sampler = unwrapSampler(this._findBoundSamplerCandidate(drawState));
      if (!sampler) {
        this._stats.bindGroupGuards += 1;
        this._stats.invalidSamplerSubstitutions += 1;
        substitutedSampler = true;
        sampler = this._ensureFallbackSampler();
        if (!sampler) {
          this._stats.bindGroupErrors += 1;
          out.substitutedCbv = substitutedCbv;
          out.substitutedTexture = substitutedTexture;
          out.substitutedSampler = true;
          return out;
        }
      }

      try {
        const bindings = this._bgScratchBindings;
        const kinds = this._bgScratchKinds;
        const resA = this._bgScratchResA;
        const resB = this._bgScratchResB;
        const genA = this._bgScratchGenA;
        const genB = this._bgScratchGenB;
        const offs = this._bgScratchOff;
        const sizes = this._bgScratchSize;
        const cbvGen = getResourceGeneration(cbvBuffer);
        const viewGen = getResourceGeneration(textureView);
        const smpGen = getResourceGeneration(sampler);
        const texSrc = sourceTexture && sourceTexture !== textureView ? sourceTexture : null;
        const srcGen = texSrc ? getResourceGeneration(texSrc) : 0;

        bindings[0] = 0;
        kinds[0] = BG_ENTRY_KIND_BUFFER;
        resA[0] = cbvBuffer;
        resB[0] = null;
        genA[0] = cbvGen;
        genB[0] = 0;
        offs[0] = cbvOffset;
        sizes[0] = cbvSize;

        bindings[1] = 1;
        kinds[1] = BG_ENTRY_KIND_TEXTURE;
        resA[1] = texSrc;
        resB[1] = textureView;
        genA[1] = srcGen;
        genB[1] = viewGen;
        offs[1] = 0;
        sizes[1] = -1;

        bindings[2] = 2;
        kinds[2] = BG_ENTRY_KIND_RAW;
        resA[2] = sampler;
        resB[2] = null;
        genA[2] = smpGen;
        genB[2] = 0;
        offs[2] = 0;
        sizes[2] = -1;

        let h = -2128831035 | 0;
        h = hashMixInt32(h, BG_OWNER_FALLBACK_SPLASH);
        h = hashMixInt32(h, this._getResourceId(pipeline));
        h = hashMixInt32(h, this._getResourceId(cbvBuffer));
        h = hashMixInt32(h, cbvGen);
        h = hashMixNum(h, cbvOffset);
        h = hashMixNum(h, cbvSize);
        h = hashMixInt32(h, this._getResourceId(textureView));
        h = hashMixInt32(h, viewGen);
        h = hashMixInt32(h, this._getResourceId(sampler));
        h = hashMixInt32(h, smpGen);

        let bindGroup = this._bgLookupNode(
          h,
          BG_OWNER_FALLBACK_SPLASH,
          pipeline,
          0,
          3,
          bindings,
          kinds,
          resA,
          resB,
          genA,
          genB,
          offs,
          sizes
        );
        let cacheHit = true;
        if (!bindGroup) {
          cacheHit = false;
          this._stats.bindGroupCacheMisses += 1;
          const layout = this._getPipelineBindGroupLayout(pipeline, 0);
          bindGroup = this._device.createBindGroup({
            label: 'draw_executor:fallback_splash_rect_bg',
            layout,
            entries: [
              { binding: 0, resource: { buffer: cbvBuffer, offset: cbvOffset, size: cbvSize } },
              { binding: 1, resource: textureView },
              { binding: 2, resource: sampler },
            ],
          });
          const tracked = texSrc
            ? [cbvBuffer, texSrc, textureView, sampler]
            : [cbvBuffer, textureView, sampler];
          this._bgStoreNode(
            h,
            BG_OWNER_FALLBACK_SPLASH,
            pipeline,
            0,
            3,
            bindings,
            kinds,
            resA,
            resB,
            genA,
            genB,
            offs,
            sizes,
            bindGroup,
            tracked
          );
        }
        resA[0] = null;
        resA[1] = null;
        resB[1] = null;
        resA[2] = null;
        const bound = this._bindBindGroupOnPass(pass, 0, bindGroup, null);
        out.ok = true;
        out.substitutedCbv = substitutedCbv;
        out.substitutedTexture = substitutedTexture;
        out.substitutedSampler = substitutedSampler;
        out.cacheHit = cacheHit;
        out.bindGroupElided = !bound;
        return out;
      } catch (err) {
        this._bgScratchResA[0] = null;
        this._bgScratchResA[1] = null;
        this._bgScratchResB[1] = null;
        this._bgScratchResA[2] = null;
        this._stats.bindGroupErrors += 1;
        this._warn('Failed to bind fallback splash rect bind group', {
          error: String(err && err.message ? err.message : err),
        });
        out.substitutedCbv = substitutedCbv;
        out.substitutedTexture = substitutedTexture;
        out.substitutedSampler = substitutedSampler;
        return out;
      }
    }

    _bindFallbackVertexBufferSplashRect24B(pass, drawState, vertexCount, firstVertex) {
      const out = this._tmpFallbackVbResult;
      out.ok = false;
      out.substitutedVertexBuffer = false;
      out.vertexBufferElided = false;
      out.effectiveVertexCount = 0;
      out.effectiveFirstVertex = 0;

      const hasSlot0 = (this._stagedVbMask & 1) !== 0;
      const fallbackSlot0 = !hasSlot0 && Array.isArray(drawState.vertexBuffers)
        ? drawState.vertexBuffers[0]
        : null;
      const requiredBytes = drawState.indexed
        ? TEXTURED_QUAD_VERTEX_STRIDE
        : (firstVertex + vertexCount) * TEXTURED_QUAD_VERTEX_STRIDE;

      const slot0Valid = hasSlot0
        ? isValidVertexBufferParts(
            this._stagedVbBuffers[0],
            this._stagedVbOffsets[0],
            this._stagedVbSizes[0],
            this._stagedVbStrides[0],
            requiredBytes,
            TEXTURED_QUAD_VERTEX_STRIDE
          )
        : fallbackSlot0
          ? isValidVertexBufferBinding(fallbackSlot0, requiredBytes, TEXTURED_QUAD_VERTEX_STRIDE)
          : false;

      if (slot0Valid) {
        const buffer = hasSlot0
          ? this._stagedVbBuffers[0]
          : fallbackSlot0.buffer !== undefined
            ? fallbackSlot0.buffer
            : fallbackSlot0;
        const offset = hasSlot0 ? this._stagedVbOffsets[0] || 0 : fallbackSlot0.offset || 0;
        const size = hasSlot0 ? this._stagedVbSizes[0] : fallbackSlot0.size;
        try {
          const bound = this._bindVertexBufferOnPass(pass, 0, buffer, offset, size);
          out.ok = true;
          out.vertexBufferElided = !bound;
          out.effectiveVertexCount = vertexCount;
          out.effectiveFirstVertex = firstVertex;
          return out;
        } catch (err) {
          this._warn(
            'setVertexBuffer threw on bound 24B splash vertex buffer; substituting synthetic splash quad',
            {
              error: String(err && err.message ? err.message : err),
            }
          );
        }
      }

      this._stats.invalidVertexBufferGuards += 1;
      this._stats.vertexBufferSubstitutions += 1;
      out.substitutedVertexBuffer = true;
      const fallbackVb = this._ensureFallbackSplashVertexBuffer24B();
      if (!fallbackVb) {
        return out;
      }

      try {
        const bound = this._bindVertexBufferOnPass(pass, 0, fallbackVb, 0, null);
        const expandedCount =
          isRectListDrawState(drawState) && vertexCount >= 3
            ? Math.floor(vertexCount / 3) * 6
            : vertexCount;
        const clampedCount = drawState.indexed ? vertexCount : Math.min(Math.max(expandedCount, 3), 6);
        out.ok = true;
        out.vertexBufferElided = !bound;
        out.effectiveVertexCount = clampedCount;
        out.effectiveFirstVertex = 0;
        return out;
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('Failed to bind synthetic 24B splash fallback vertex buffer', {
          error: String(err && err.message ? err.message : err),
        });
        return out;
      }
    }

    _bindNormalResources(pass, result) {
      this._syncPassTracking(pass);
      let vertexBuffersElided = 0;
      let bindGroupsElided = 0;

      // 1. Fast O(1) bitmask vertex-buffer binding for slots 0..7 (zero Map/object churn).
      let vbMask = this._stagedVbMask;
      if (vbMask !== 0) {
        const elideVb = this._elideRedundantVertexBuffers;
        const dirtyVb = this._dirtyVbMask;
        const boundVb = this._boundVbMask;
        const vbBufs = this._stagedVbBuffers;
        const vbOffs = this._stagedVbOffsets;
        const vbSizes = this._stagedVbSizes;
        const vbStrides = this._stagedVbStrides;

        while (vbMask !== 0) {
          const bit = vbMask & -vbMask;
          const slot = 31 - Math.clz32(bit);
          vbMask ^= bit;
          const buf = vbBufs[slot];

          if (
            elideVb &&
            (dirtyVb & bit) === 0 &&
            (boundVb & bit) !== 0 &&
            buf !== null &&
            buf.destroyed !== true
          ) {
            vertexBuffersElided += 1;
            continue;
          }

          const off = vbOffs[slot];
          const sz = vbSizes[slot];
          if (!isValidVertexBufferParts(buf, off, sz, vbStrides[slot], 0, null)) {
            this._boundVbMask &= ~bit;
            this._boundVbBuffers[slot] = null;
            this._boundVbOffsets[slot] = 0;
            this._boundVbSizes[slot] = null;
            this._dirtyVbMask |= bit;
            this._stats.invalidVertexBufferGuards += 1;
            this._stats.skippedVertexBufferBindings += 1;
            continue;
          }
          try {
            const bound = this._bindFastVertexBufferOnPassSynced(
              pass,
              slot,
              bit,
              buf,
              off || 0,
              sz
            );
            if (!bound) {
              vertexBuffersElided += 1;
            }
          } catch (err) {
            this._stats.passErrors += 1;
            this._stats.invalidVertexBufferGuards += 1;
            this._stats.skippedVertexBufferBindings += 1;
            this._warn('setVertexBuffer failed on normal draw', {
              slot,
              error: String(err && err.message ? err.message : err),
            });
          }
        }
      }

      if (vertexBuffersElided > 0) {
        this._stats.redundantVertexBufferSkips += vertexBuffersElided;
        this._stats.elidedVertexBufferSets += vertexBuffersElided;
        this._stats.totalElidedStateCalls += vertexBuffersElided;
        this._totalStateCalls += vertexBuffersElided;
      }

      if (this._stagedVbOverflow !== null && this._stagedVbOverflow.size > 0) {
        for (const [slot, entry] of this._stagedVbOverflow.entries()) {
          if (!isValidVertexBufferBinding(entry, 0, null)) {
            if (this._boundVbOverflow !== null) {
              this._boundVbOverflow.delete(slot);
            }
            this._stats.invalidVertexBufferGuards += 1;
            this._stats.skippedVertexBufferBindings += 1;
            continue;
          }
          try {
            const bound = this._bindVertexBufferOnPass(
              pass,
              slot,
              entry.buffer,
              entry.offset || 0,
              entry.size
            );
            if (!bound) {
              vertexBuffersElided += 1;
            }
          } catch (err) {
            this._stats.passErrors += 1;
            this._stats.invalidVertexBufferGuards += 1;
            this._stats.skippedVertexBufferBindings += 1;
            this._warn('setVertexBuffer failed on normal draw', {
              slot,
              error: String(err && err.message ? err.message : err),
            });
          }
        }
      }

      // 2. Fast O(1) bitmask bind-group binding for slots 0..3 (zero Map/object/string/array churn).
      let bgMask = this._stagedBgMask;
      if (bgMask !== 0) {
        const elideBg = this._elideRedundantBindGroups;
        const dirtyBg = this._dirtyBgMask;
        const boundBg = this._boundBgMask;
        const bgGroups = this._stagedBgGroups;
        const bgHasDyn = this._stagedBgHasDyn;
        const bgDynSrc = this._stagedBgDynSource;
        const bgDynStart = this._stagedBgDynStart;
        const bgDynLen = this._stagedBgDynLength;

        while (bgMask !== 0) {
          const bit = bgMask & -bgMask;
          const slot = 31 - Math.clz32(bit);
          bgMask ^= bit;
          const bg = bgGroups[slot];
          if (!bg || bg.destroyed === true) {
            this._boundBgMask &= ~bit;
            this._boundBgGroups[slot] = null;
            this._boundBgHasDyn[slot] = 0;
            this._boundBgDynLen[slot] = 0;
            this._dirtyBgMask |= bit;
            this._stats.bindGroupGuards += 1;
            this._stats.skippedBindGroupBindings += 1;
            continue;
          }

          const hasDyn = bgHasDyn[slot];
          if (
            elideBg &&
            (dirtyBg & bit) === 0 &&
            (boundBg & bit) !== 0 &&
            (hasDyn === 0 ||
              this._fastDynamicOffsetsEqual(
                slot,
                1,
                bgDynSrc[slot],
                bgDynStart[slot],
                bgDynLen[slot]
              ))
          ) {
            bindGroupsElided += 1;
            continue;
          }

          try {
            const bound = this._bindFastBindGroupOnPassSynced(
              pass,
              slot,
              bit,
              bg,
              hasDyn,
              bgDynSrc[slot],
              bgDynStart[slot],
              bgDynLen[slot]
            );
            if (!bound) {
              bindGroupsElided += 1;
            }
          } catch (err) {
            this._stats.passErrors += 1;
            this._stats.bindGroupGuards += 1;
            this._stats.skippedBindGroupBindings += 1;
            this._warn('setBindGroup failed on normal draw', {
              groupIndex: slot,
              error: String(err && err.message ? err.message : err),
            });
          }
        }
      }

      if (bindGroupsElided > 0) {
        this._stats.redundantBindGroupSkips += bindGroupsElided;
        this._stats.elidedBindGroupSets += bindGroupsElided;
        this._stats.totalElidedStateCalls += bindGroupsElided;
        this._totalStateCalls += bindGroupsElided;
      }

      if (this._stagedBgOverflow !== null && this._stagedBgOverflow.size > 0) {
        for (const [groupIndex, entry] of this._stagedBgOverflow.entries()) {
          if (!entry || !entry.bindGroup || entry.bindGroup.destroyed === true) {
            if (this._boundBgOverflow !== null) {
              this._boundBgOverflow.delete(groupIndex);
            }
            this._stats.bindGroupGuards += 1;
            this._stats.skippedBindGroupBindings += 1;
            continue;
          }
          try {
            const bound = this._bindBindGroupOnPass(
              pass,
              groupIndex,
              entry.bindGroup,
              entry.dynamicOffsets || null
            );
            if (!bound) {
              bindGroupsElided += 1;
            }
          } catch (err) {
            this._stats.passErrors += 1;
            this._stats.bindGroupGuards += 1;
            this._stats.skippedBindGroupBindings += 1;
            this._warn('setBindGroup failed on normal draw', {
              groupIndex,
              error: String(err && err.message ? err.message : err),
            });
          }
        }
      }

      result.vertexBuffersElided = vertexBuffersElided;
      result.bindGroupsElided = bindGroupsElided;
    }

    _validateAndBindIndexBuffer(pass, indexCount, firstIndex) {
      if (!this._hasStagedIb) {
        this._hasBoundIb = false;
        this._boundIbBuffer = null;
        return IB_STATUS_INVALID;
      }
      const buf = this._stagedIbBuffer;
      const format =
        this._stagedIbFormat === 'uint32'
          ? 'uint32'
          : this._stagedIbFormat === 'uint16'
            ? 'uint16'
            : null;
      if (!format || !isValidGpuBuffer(buf, BUFFER_USAGE.INDEX)) {
        this._hasBoundIb = false;
        this._boundIbBuffer = null;
        return IB_STATUS_INVALID;
      }
      const elemBytes = format === 'uint32' ? 4 : 2;
      const offset = this._stagedIbOffset || 0;
      if (!Number.isInteger(offset) || offset < 0 || offset % elemBytes !== 0) {
        return IB_STATUS_INVALID_RANGE;
      }
      const bufSize = typeof buf.size === 'number' && Number.isFinite(buf.size) ? buf.size : null;
      if (bufSize !== null) {
        const sz = this._stagedIbSize;
        const effectiveSize = sz !== undefined && sz !== null ? sz : bufSize - offset;
        if (offset >= bufSize || offset + effectiveSize > bufSize) {
          return IB_STATUS_INVALID_RANGE;
        }
        const neededBytes = (firstIndex + indexCount) * elemBytes;
        if (effectiveSize < neededBytes) {
          return IB_STATUS_INVALID_RANGE;
        }
      }
      if (this._elideRedundantIndexBuffers && !this._dirtyIb && this._hasBoundIb) {
        this._stats.redundantIndexBufferSkips += 1;
        this._stats.elidedIndexBufferSets += 1;
        this._stats.totalElidedStateCalls += 1;
        this._totalStateCalls += 1;
        return IB_STATUS_ELIDED;
      }
      try {
        const bound = this._bindIndexBufferOnPass(pass, buf, format, offset, this._stagedIbSize);
        return bound ? IB_STATUS_EMITTED : IB_STATUS_ELIDED;
      } catch (err) {
        this._stats.passErrors += 1;
        this._warn('setIndexBuffer failed', {
          error: String(err && err.message ? err.message : err),
        });
        return IB_STATUS_INVALID;
      }
    }

    _acquireDrawResult(indexed, indirect, count, instanceCount, indirectOffset, reuseOverride) {
      if (this._reuseDrawResult || reuseOverride === true) {
        return resetDrawResultInPlace(
          indirect ? this._reusableIndirectResult : this._reusableDrawResult,
          indexed,
          indirect,
          count,
          instanceCount,
          indirectOffset
        );
      }
      return createDrawResultObject(indexed, indirect, count, instanceCount, indirectOffset);
    }

    /**
     * Execute a WebGPU compute dispatch (`dispatchWorkgroups` or `dispatchWorkgroupsIndirect`)
     * with redundant `setPipeline` / `setBindGroup` elision and `(pipeline, groupIndex, entries)`
     * bind-group caching (avoiding per-dispatch `device.createBindGroup()` when bindings repeat).
     *
     * @param {object} pass GPUComputePassEncoder
     * @param {{
     *   pipeline?: object,
     *   bindGroups?: Array<object>,
     *   workgroupCountX?: number,
     *   workgroupCountY?: number,
     *   workgroupCountZ?: number,
     *   x?: number,
     *   y?: number,
     *   z?: number,
     *   indirect?: boolean,
     *   indirectBuffer?: object,
     *   indirectOffset?: number,
     *   reuseResult?: boolean,
     * }} [computeArgs]
     */
    executeCompute(pass, computeArgs = {}) {
      const args = computeArgs && typeof computeArgs === 'object' ? computeArgs : null;
      const indirect = Boolean(args && (args.indirect === true || args.indirectBuffer));
      if (
        indirect &&
        args &&
        args.clamp === true &&
        (args.commandEncoder || args.clampComputePass)
      ) {
        return this.executeClampedDispatchIndirect(
          args.commandEncoder || null,
          pass,
          args
        );
      }
      const x = args
        ? toNonNegativeInt(args.workgroupCountX !== undefined ? args.workgroupCountX : args.x, 0)
        : 0;
      const y = args
        ? toNonNegativeInt(
            args.workgroupCountY !== undefined
              ? args.workgroupCountY
              : args.y !== undefined
                ? args.y
                : 1,
            0
          )
        : 1;
      const z = args
        ? toNonNegativeInt(
            args.workgroupCountZ !== undefined
              ? args.workgroupCountZ
              : args.z !== undefined
                ? args.z
                : 1,
            0
          )
        : 1;
      const indirectBuffer = args && args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const indirectOffset = args && args.indirectOffset !== undefined ? args.indirectOffset : 0;
      const reuse = this._reuseDrawResult || (args !== null && args.reuseResult === true);
      const result = reuse
        ? this._reusableComputeResult
        : {
            executed: false,
            skipped: false,
            reason: null,
            indirect: false,
            indirectOffset: 0,
            workgroupCountX: 0,
            workgroupCountY: 0,
            workgroupCountZ: 0,
            pipelineElided: false,
            bindGroupsElided: 0,
            bindGroupCacheHits: 0,
            bindGroupCacheMisses: 0,
          };

      result.executed = false;
      result.skipped = false;
      result.reason = null;
      result.indirect = indirect;
      result.indirectOffset =
        indirect && Number.isInteger(indirectOffset) && indirectOffset >= 0 ? indirectOffset : 0;
      result.workgroupCountX = indirect ? 0 : x;
      result.workgroupCountY = indirect ? 0 : y;
      result.workgroupCountZ = indirect ? 0 : z;
      result.pipelineElided = false;
      result.bindGroupsElided = 0;
      result.bindGroupCacheHits = 0;
      result.bindGroupCacheMisses = 0;

      if (this._destroyed) {
        this._stats.skippedComputeDispatches += 1;
        result.skipped = true;
        result.reason = 'executor-destroyed';
        return result;
      }

      const dispatchMethod = indirect ? 'dispatchWorkgroupsIndirect' : 'dispatchWorkgroups';
      if (
        !pass ||
        typeof pass !== 'object' ||
        typeof pass.setPipeline !== 'function' ||
        typeof pass[dispatchMethod] !== 'function'
      ) {
        this._stats.skippedComputeDispatches += 1;
        result.skipped = true;
        result.reason = 'invalid-compute-pass';
        return result;
      }

      if (indirect) {
        if (validateComputeIndirectBufferStatus(indirectBuffer, indirectOffset) !== INDIRECT_STATUS_OK) {
          this._stats.skippedComputeDispatches += 1;
          result.skipped = true;
          result.reason = 'invalid-indirect-buffer';
          return result;
        }
      } else if (x <= 0 || y <= 0 || z <= 0) {
        this._stats.skippedComputeDispatches += 1;
        result.skipped = true;
        result.reason = 'zero-workgroup-count';
        return result;
      }

      const rawPipeline =
        args !== null && args.pipeline !== undefined ? args.pipeline : this._activeComputePipeline;
      const computePipeline =
        rawPipeline && typeof rawPipeline === 'object' && rawPipeline.pipeline
          ? rawPipeline.pipeline
          : rawPipeline;
      if (!computePipeline || typeof computePipeline !== 'object' || computePipeline.destroyed === true) {
        this._stats.skippedComputeDispatches += 1;
        result.skipped = true;
        result.reason = 'no-compute-pipeline';
        return result;
      }

      try {
        const pipeBound = this._bindPipelineOnPass(pass, computePipeline);
        result.pipelineElided = !pipeBound;

        let bindGroupsElided = 0;
        const explicitBindGroups = args !== null && Array.isArray(args.bindGroups) ? args.bindGroups : null;

        if (explicitBindGroups !== null) {
          for (let i = 0; i < explicitBindGroups.length; i += 1) {
            const spec = explicitBindGroups[i];
            if (!spec || typeof spec !== 'object') {
              continue;
            }
            const groupIndex = Number.isInteger(spec.groupIndex) ? spec.groupIndex : i;
            let bg = null;
            if (Array.isArray(spec.entries) && !spec.bindGroup && spec.kind !== 'GPUBindGroup') {
              const hitsBefore = this._stats.bindGroupCacheHits;
              const missesBefore = this._stats.bindGroupCacheMisses;
              const specLabel = typeof spec.label === 'string' ? spec.label : null;
              if (spec.layout) {
                bg =
                  this._device && typeof this._device.createBindGroup === 'function'
                    ? this._getOrCreateBindGroupStructured(
                        BG_OWNER_LAYOUT,
                        spec.layout,
                        groupIndex >= 0 ? groupIndex : -1,
                        spec.entries,
                        specLabel
                      )
                    : ((this._stats.bindGroupErrors += 1), null);
              } else {
                bg =
                  this._device &&
                  typeof this._device.createBindGroup === 'function' &&
                  typeof computePipeline.getBindGroupLayout === 'function' &&
                  groupIndex >= 0
                    ? this._getOrCreateBindGroupStructured(
                        BG_OWNER_PIPELINE,
                        computePipeline,
                        groupIndex,
                        spec.entries,
                        specLabel
                      )
                    : ((this._stats.bindGroupErrors += 1), null);
              }
              result.bindGroupCacheHits += this._stats.bindGroupCacheHits - hitsBefore;
              result.bindGroupCacheMisses += this._stats.bindGroupCacheMisses - missesBefore;
              if (!bg) {
                this._stats.skippedComputeDispatches += 1;
                result.skipped = true;
                result.reason = 'compute-bindgroup-failed';
                return result;
              }
            } else {
              bg = spec.bindGroup !== undefined ? spec.bindGroup : spec;
            }
            if (!bg || bg.destroyed === true) {
              this._stats.bindGroupGuards += 1;
              this._stats.skippedBindGroupBindings += 1;
              continue;
            }
            const dynOffsets = spec.dynamicOffsets !== undefined ? spec.dynamicOffsets : null;
            const bound = this._bindBindGroupOnPass(pass, groupIndex, bg, dynOffsets);
            if (!bound) {
              bindGroupsElided += 1;
            }
          }
        } else {
          // Bind any staged bind groups (fast bitmask slots 0..3 + overflow).
          let bgMask = this._stagedBgMask;
          while (bgMask !== 0) {
            const bit = bgMask & -bgMask;
            const slot = 31 - Math.clz32(bit);
            bgMask ^= bit;
            const bg = this._stagedBgGroups[slot];
            if (!bg || bg.destroyed === true) {
              this._stats.bindGroupGuards += 1;
              this._stats.skippedBindGroupBindings += 1;
              continue;
            }
            const bound = this._bindFastBindGroupOnPassSynced(
              pass,
              slot,
              bit,
              bg,
              this._stagedBgHasDyn[slot],
              this._stagedBgDynSource[slot],
              this._stagedBgDynStart[slot],
              this._stagedBgDynLength[slot]
            );
            if (!bound) {
              bindGroupsElided += 1;
              this._stats.redundantBindGroupSkips += 1;
              this._stats.elidedBindGroupSets += 1;
              this._stats.totalElidedStateCalls += 1;
              this._totalStateCalls += 1;
            }
          }
          if (this._stagedBgOverflow !== null && this._stagedBgOverflow.size > 0) {
            for (const [groupIndex, entry] of this._stagedBgOverflow.entries()) {
              if (!entry || !entry.bindGroup || entry.bindGroup.destroyed === true) {
                this._stats.bindGroupGuards += 1;
                this._stats.skippedBindGroupBindings += 1;
                continue;
              }
              const bound = this._bindBindGroupOnPass(
                pass,
                groupIndex,
                entry.bindGroup,
                entry.dynamicOffsets || null
              );
              if (!bound) {
                bindGroupsElided += 1;
              }
            }
          }
        }

        result.bindGroupsElided = bindGroupsElided;
        if (indirect) {
          pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
        } else {
          pass.dispatchWorkgroups(x, y, z);
        }
        this._stats.computeDispatches += 1;
        result.executed = true;
        return result;
      } catch (err) {
        this._stats.passErrors += 1;
        this._stats.skippedComputeDispatches += 1;
        this._warn('Compute dispatch execution failed', {
          error: String(err && err.message ? err.message : err),
        });
        result.skipped = true;
        result.reason = 'pass-error';
        return result;
      }
    }

    /**
     * Positional convenience wrapper for `executeCompute` (`dispatchWorkgroups`).
     */
    dispatchWorkgroups(pass, workgroupCountX, workgroupCountY = 1, workgroupCountZ = 1) {
      return this.executeCompute(pass, {
        workgroupCountX,
        workgroupCountY,
        workgroupCountZ,
      });
    }

    /**
     * Positional convenience wrapper for `executeCompute` (`dispatchWorkgroupsIndirect`).
     */
    dispatchWorkgroupsIndirect(pass, indirectBuffer, indirectOffset = 0) {
      return this.executeCompute(pass, {
        indirect: true,
        indirectBuffer,
        indirectOffset,
      });
    }

    /**
     * Encode a GPU-side compute clamp pass for `draw`, `drawIndexed`, `dispatch`, or
     * `dispatchMesh` indirect argument buffers without CPU readbacks.
     * Pass a preallocated `out` object for zero per-draw GC allocations.
     *
     * @param {object} passOrEncoder GPUCommandEncoder or GPUComputePassEncoder
     * @param {string} kind 'draw' | 'drawIndexed' | 'dispatch' | 'dispatchMesh'
     * @param {object} indirectBuffer Source GPU-written indirect buffer
     * @param {number} [indirectOffset=0]
     * @param {object} [options]
     * @param {object} [out]
     */
    clampIndirectArgs(passOrEncoder, kind, indirectBuffer, indirectOffset = 0, options, out) {
      const target =
        out !== undefined && out !== null && typeof out === 'object'
          ? out
          : options && options.reuseResult === true
            ? this._scratchClampResult
            : null;
      const clampExec =
        (options && options.clampExecutor) || this.getIndirectClampExecutor();
      if (!clampExec) {
        if (target) {
          target.ok = false;
          target.encoded = false;
          target.kind = kind;
          target.commandCount = 0;
          target.workgroupsX = 0;
          target.dstBuffer = null;
          target.dstOffset = 0;
          target.dstByteLength = 0;
          target.recordBytes = 0;
          target.stride = 0;
          target.zeroedFirstInstance = false;
          target.hasCountBuffer = false;
          target.cacheHit = false;
          target.reason = 'clamp-executor-unavailable';
          return target;
        }
        return {
          ok: false,
          encoded: false,
          kind,
          commandCount: 0,
          workgroupsX: 0,
          dstBuffer: null,
          dstOffset: 0,
          dstByteLength: 0,
          recordBytes: 0,
          stride: 0,
          zeroedFirstInstance: false,
          hasCountBuffer: false,
          cacheHit: false,
          reason: 'clamp-executor-unavailable',
        };
      }

      const opts = this._scratchClampOptions;
      opts.srcBuffer = indirectBuffer;
      opts.srcOffset = indirectOffset;
      opts.dstBuffer = options && options.dstBuffer !== undefined ? options.dstBuffer : null;
      opts.dstOffset = options && options.dstOffset !== undefined ? options.dstOffset : 0;
      opts.countBuffer = options && options.countBuffer !== undefined ? options.countBuffer : null;
      opts.countOffset = options && options.countOffset !== undefined ? options.countOffset : 0;
      opts.commandCount =
        options && options.commandCount !== undefined
          ? options.commandCount
          : options && options.count !== undefined
            ? options.count
            : 1;
      opts.stride = options && options.stride !== undefined ? options.stride : undefined;
      opts.maxVertexCount =
        options && options.maxVertexCount !== undefined ? options.maxVertexCount : undefined;
      opts.maxIndexCount =
        options && options.maxIndexCount !== undefined ? options.maxIndexCount : undefined;
      opts.maxInstanceCount =
        options && options.maxInstanceCount !== undefined ? options.maxInstanceCount : undefined;
      opts.maxWorkgroupsX =
        options && options.maxWorkgroupsX !== undefined ? options.maxWorkgroupsX : undefined;
      opts.maxWorkgroupsY =
        options && options.maxWorkgroupsY !== undefined ? options.maxWorkgroupsY : undefined;
      opts.maxWorkgroupsZ =
        options && options.maxWorkgroupsZ !== undefined ? options.maxWorkgroupsZ : undefined;
      opts.maxMeshVerticesPerGroup =
        options && options.maxMeshVerticesPerGroup !== undefined
          ? options.maxMeshVerticesPerGroup
          : undefined;
      opts.maxMeshTotalVertices =
        options && options.maxMeshTotalVertices !== undefined
          ? options.maxMeshTotalVertices
          : undefined;
      opts.indirectFirstInstance =
        options && options.indirectFirstInstance !== undefined
          ? options.indirectFirstInstance
          : this.supportsIndirectFirstInstance();
      opts.maxIndirectBatchCount =
        options && options.maxIndirectBatchCount !== undefined
          ? options.maxIndirectBatchCount
          : undefined;
      opts.clampBatch = !options || options.clampBatch !== false;

      const res = clampExec.encodeClamp(passOrEncoder, kind, opts, target);
      if (res.ok && res.encoded) {
        this._stats.indirectClampPasses += 1;
        if (res.zeroedFirstInstance) {
          this._stats.indirectFirstInstanceClamped += res.commandCount;
        }
      }
      return res;
    }

    /**
     * Execute a GPU-clamped `drawIndirect` (or `drawIndexedIndirect` when `drawArgs.indexed`
     * is true): runs the compute clamp pass on `encoder` (or `drawArgs.clampComputePass`),
     * then executes the indirect render draw on `renderPass` from the validated output buffer.
     * Zero per-draw GC allocations when `reuseResult: true` is set or `this._reuseDrawResult` is true.
     */
    executeClampedDrawIndirect(encoder, renderPass, drawArgs = {}) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : null;
      const indexed = Boolean(args && args.indexed);
      const srcBuf = args && args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const srcOff = args && args.indirectOffset !== undefined ? args.indirectOffset : 0;
      const clampTarget = args && args.clampComputePass ? args.clampComputePass : encoder;

      const clampRes = this.clampIndirectArgs(
        clampTarget,
        indexed ? 'drawIndexed' : 'draw',
        srcBuf,
        srcOff,
        args,
        this._scratchClampResult
      );
      if (!clampRes.ok) {
        const result = this._acquireDrawResult(
          indexed,
          true,
          0,
          0,
          Number.isInteger(srcOff) && srcOff >= 0 ? srcOff : 0,
          args !== null ? args.reuseResult : false
        );
        this._recordSkip('invalid-indirect-buffer', true);
        result.skipped = true;
        result.reason = clampRes.reason || 'indirect-clamp-failed';
        return result;
      }
      if (!clampRes.encoded) {
        const result = this._acquireDrawResult(
          indexed,
          true,
          0,
          0,
          clampRes.dstOffset,
          args !== null ? args.reuseResult : false
        );
        this._recordSkip('zero-draw-count', false);
        result.skipped = true;
        result.reason = 'zero-draw-count';
        return result;
      }

      const drawRes = this._executeDrawIndirectCore(
        renderPass,
        args,
        indexed,
        clampRes.dstBuffer,
        clampRes.dstOffset
      );
      if (drawRes.executed) {
        this._stats.clampedIndirectDraws += 1;
      }
      drawRes.clamped = true;
      drawRes.clampedBuffer = clampRes.dstBuffer;
      drawRes.clampedOffset = clampRes.dstOffset;
      drawRes.zeroedFirstInstance = clampRes.zeroedFirstInstance;
      return drawRes;
    }

    /**
     * Convenience wrapper for GPU-clamped `drawIndexedIndirect` (20B record).
     */
    executeClampedDrawIndexedIndirect(encoder, renderPass, drawArgs = {}) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : null;
      const srcBuf = args && args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const srcOff = args && args.indirectOffset !== undefined ? args.indirectOffset : 0;
      const clampTarget = args && args.clampComputePass ? args.clampComputePass : encoder;

      const clampRes = this.clampIndirectArgs(
        clampTarget,
        'drawIndexed',
        srcBuf,
        srcOff,
        args,
        this._scratchClampResult
      );
      if (!clampRes.ok) {
        const result = this._acquireDrawResult(
          true,
          true,
          0,
          0,
          Number.isInteger(srcOff) && srcOff >= 0 ? srcOff : 0,
          args !== null ? args.reuseResult : false
        );
        this._recordSkip('invalid-indirect-buffer', true);
        result.skipped = true;
        result.reason = clampRes.reason || 'indirect-clamp-failed';
        return result;
      }
      if (!clampRes.encoded) {
        const result = this._acquireDrawResult(
          true,
          true,
          0,
          0,
          clampRes.dstOffset,
          args !== null ? args.reuseResult : false
        );
        this._recordSkip('zero-draw-count', false);
        result.skipped = true;
        result.reason = 'zero-draw-count';
        return result;
      }

      const drawRes = this._executeDrawIndirectCore(
        renderPass,
        args,
        true,
        clampRes.dstBuffer,
        clampRes.dstOffset
      );
      if (drawRes.executed) {
        this._stats.clampedIndirectDraws += 1;
      }
      drawRes.clamped = true;
      drawRes.clampedBuffer = clampRes.dstBuffer;
      drawRes.clampedOffset = clampRes.dstOffset;
      drawRes.zeroedFirstInstance = clampRes.zeroedFirstInstance;
      return drawRes;
    }

    /**
     * Execute a GPU-clamped `dispatchWorkgroupsIndirect` (12B record): clamps workgroup
     * X/Y/Z against `device.limits.maxComputeWorkgroupsPerDimension` (+ optional count buffer)
     * before dispatching on `computePass` (or an encoder-opened compute pass).
     */
    executeClampedDispatchIndirect(encoder, computePass, computeArgs = {}) {
      const args = computeArgs && typeof computeArgs === 'object' ? computeArgs : null;
      const srcBuf = args && args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const srcOff = args && args.indirectOffset !== undefined ? args.indirectOffset : 0;
      const clampTarget = args && args.clampComputePass ? args.clampComputePass : encoder;

      const clampRes = this.clampIndirectArgs(
        clampTarget,
        'dispatch',
        srcBuf,
        srcOff,
        args,
        this._scratchClampResult
      );
      if (!clampRes.ok) {
        this._stats.skippedComputeDispatches += 1;
        const reuse = this._reuseDrawResult || (args !== null && args.reuseResult === true);
        const r = reuse
          ? this._reusableComputeResult
          : {
              executed: false,
              skipped: true,
              reason: null,
              indirect: true,
              indirectOffset: 0,
              workgroupCountX: 0,
              workgroupCountY: 0,
              workgroupCountZ: 0,
              pipelineElided: false,
              bindGroupsElided: 0,
              bindGroupCacheHits: 0,
              bindGroupCacheMisses: 0,
            };
        r.executed = false;
        r.skipped = true;
        r.reason = clampRes.reason || 'indirect-clamp-failed';
        r.indirect = true;
        r.indirectOffset = 0;
        r.clamped = false;
        return r;
      }
      if (!clampRes.encoded) {
        this._stats.skippedComputeDispatches += 1;
        const reuse = this._reuseDrawResult || (args !== null && args.reuseResult === true);
        const r = reuse
          ? this._reusableComputeResult
          : {
              executed: false,
              skipped: true,
              reason: null,
              indirect: true,
              indirectOffset: 0,
              workgroupCountX: 0,
              workgroupCountY: 0,
              workgroupCountZ: 0,
              pipelineElided: false,
              bindGroupsElided: 0,
              bindGroupCacheHits: 0,
              bindGroupCacheMisses: 0,
            };
        r.executed = false;
        r.skipped = true;
        r.reason = 'zero-workgroup-count';
        r.indirect = true;
        r.indirectOffset = clampRes.dstOffset;
        r.clamped = false;
        return r;
      }

      const ownedPass = !computePass;
      const targetPass =
        computePass ||
        (encoder && typeof encoder.beginComputePass === 'function'
          ? encoder.beginComputePass(args && args.computePassDescriptor)
          : null);

      const cArgs = this._scratchMeshComputeArgs;
      cArgs.pipeline = args && args.pipeline !== undefined ? args.pipeline : this._activeComputePipeline;
      cArgs.bindGroups = args && args.bindGroups !== undefined ? args.bindGroups : null;
      cArgs.workgroupCountX = 0;
      cArgs.workgroupCountY = 0;
      cArgs.workgroupCountZ = 0;
      cArgs.indirect = true;
      cArgs.indirectBuffer = clampRes.dstBuffer;
      cArgs.indirectOffset = clampRes.dstOffset;
      cArgs.reuseResult = Boolean(this._reuseDrawResult || (args !== null && args.reuseResult === true));

      const execRes = this.executeCompute(targetPass, cArgs);
      if (ownedPass && targetPass && typeof targetPass.end === 'function') {
        targetPass.end();
      }
      if (execRes.executed) {
        this._stats.clampedIndirectDispatches += 1;
      }
      execRes.clamped = true;
      execRes.clampedBuffer = clampRes.dstBuffer;
      execRes.clampedOffset = clampRes.dstOffset;
      return execRes;
    }

    /**
     * Orchestrate a lowered mesh-shader compute-then-draw pipeline (#1537 / #1552):
     * 1. Runs the lowered mesh compute shader (direct `dispatchWorkgroups` or indirect
     *    `dispatchWorkgroupsIndirect` with optional GPU clamping).
     * 2. Optionally clamps the mesh-generated `drawIndirect` / `drawIndexedIndirect` buffer.
     * 3. Executes the render pass draw(s) over the generated mesh geometry buffers.
     *
     * Zero per-draw heap allocations when `out` (or `meshSpec.out` / `reuseResult: true`) is used.
     *
     * @param {object} commandEncoder GPUCommandEncoder (optional if caller passes both computePass and renderPass)
     * @param {object} meshSpec
     * @param {object} [out]
     */
    executeMeshComputeDraw(commandEncoder, meshSpec, out) {
      const spec = meshSpec && typeof meshSpec === 'object' ? meshSpec : null;
      const reuse =
        this._reuseDrawResult || (spec !== null && spec.reuseResult === true);
      const result =
        out !== undefined && out !== null && typeof out === 'object'
          ? out
          : spec && spec.out && typeof spec.out === 'object'
            ? spec.out
            : reuse
              ? this._reusableMeshResult
              : {
                  ok: false,
                  executed: false,
                  skipped: false,
                  elided: false,
                  reason: null,
                  indirectDispatch: false,
                  dispatchClamped: false,
                  drawClamped: false,
                  indexed: false,
                  drawsIssued: 0,
                  drawIndirectBuffer: null,
                  drawIndirectOffset: 0,
                  zeroedFirstInstance: false,
                };

      result.ok = false;
      result.executed = false;
      result.skipped = false;
      result.elided = false;
      result.reason = null;
      result.indirectDispatch = false;
      result.dispatchClamped = false;
      result.drawClamped = false;
      result.indexed = false;
      result.drawsIssued = 0;
      result.drawIndirectBuffer = null;
      result.drawIndirectOffset = 0;
      result.zeroedFirstInstance = false;

      if (this._destroyed) {
        this._recordSkip('executor-destroyed', false);
        result.skipped = true;
        result.reason = 'executor-destroyed';
        return result;
      }
      if (!spec) {
        this._recordSkip('invalid-mesh-spec', false);
        result.skipped = true;
        result.reason = 'invalid-mesh-spec';
        return result;
      }

      const indexed = Boolean(spec.indexed);
      const indirectDispatch = Boolean(
        spec.indirectDispatch === true || spec.dispatchIndirectBuffer
      );
      result.indexed = indexed;
      result.indirectDispatch = indirectDispatch;

      const drawCount =
        spec.drawCount !== undefined ? toNonNegativeInt(spec.drawCount, -1) : 1;
      if (drawCount < 0) {
        this._recordSkip('invalid-draw-count', true);
        result.skipped = true;
        result.reason = 'invalid-draw-count';
        return result;
      }
      if (drawCount === 0) {
        this._stats.meshZeroCountElided += 1;
        result.ok = true;
        result.elided = true;
        result.skipped = true;
        result.reason = 'zero-draw-count';
        return result;
      }

      let gx = 0;
      let gy = 1;
      let gz = 1;
      if (!indirectDispatch) {
        gx = toNonNegativeInt(
          spec.workgroupCountX !== undefined
            ? spec.workgroupCountX
            : spec.meshGroupsX !== undefined
              ? spec.meshGroupsX
              : spec.x,
          0
        );
        gy = toNonNegativeInt(
          spec.workgroupCountY !== undefined
            ? spec.workgroupCountY
            : spec.meshGroupsY !== undefined
              ? spec.meshGroupsY
              : spec.y !== undefined
                ? spec.y
                : 1,
          0
        );
        gz = toNonNegativeInt(
          spec.workgroupCountZ !== undefined
            ? spec.workgroupCountZ
            : spec.meshGroupsZ !== undefined
              ? spec.meshGroupsZ
              : spec.z !== undefined
                ? spec.z
                : 1,
          0
        );
        if (gx <= 0 || gy <= 0 || gz <= 0) {
          this._stats.meshZeroCountElided += 1;
          result.ok = true;
          result.elided = true;
          result.skipped = true;
          result.reason = 'zero-workgroup-count';
          return result;
        }
        const maxWg =
          this._maxComputeWorkgroupsOverride !== null
            ? this._maxComputeWorkgroupsOverride
            : this._device &&
                this._device.limits &&
                typeof this._device.limits.maxComputeWorkgroupsPerDimension === 'number' &&
                this._device.limits.maxComputeWorkgroupsPerDimension >= 1
              ? Math.floor(this._device.limits.maxComputeWorkgroupsPerDimension)
              : DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
        if (gx > maxWg) gx = maxWg;
        if (gy > maxWg) gy = maxWg;
        if (gz > maxWg) gz = maxWg;
      }

      const meshPipeline =
        spec.meshPipeline !== undefined ? spec.meshPipeline : this._activeComputePipeline;
      if (!meshPipeline || typeof meshPipeline !== 'object' || meshPipeline.destroyed === true) {
        this._stats.skippedComputeDispatches += 1;
        this._recordSkip('no-mesh-pipeline', false);
        result.skipped = true;
        result.reason = 'no-mesh-pipeline';
        return result;
      }

      const renderPipeline =
        spec.renderPipeline !== undefined ? spec.renderPipeline : this._activePipeline;

      // Stage 1: Optional GPU clamp of indirect mesh dispatch buffer.
      let dispatchBuf = spec.dispatchIndirectBuffer || null;
      let dispatchOff = spec.dispatchIndirectOffset || 0;
      const canClamp = Boolean(spec.clampExecutor || this.getIndirectClampExecutor());

      if (indirectDispatch && spec.clampDispatch !== false && canClamp) {
        const clampTarget = spec.clampComputePass || commandEncoder;
        if (
          clampTarget &&
          (typeof clampTarget.beginComputePass === 'function' ||
            typeof clampTarget.dispatchWorkgroups === 'function')
        ) {
          const clampOpts = this._scratchClampOptions;
          clampOpts.srcBuffer = dispatchBuf;
          clampOpts.srcOffset = dispatchOff;
          clampOpts.dstBuffer = spec.clampedDispatchBuffer || null;
          clampOpts.dstOffset = spec.clampedDispatchOffset || 0;
          clampOpts.countBuffer = spec.dispatchCountBuffer || null;
          clampOpts.countOffset = spec.dispatchCountOffset || 0;
          clampOpts.commandCount = 1;
          clampOpts.stride = 12;
          clampOpts.maxVertexCount = undefined;
          clampOpts.maxIndexCount = undefined;
          clampOpts.maxInstanceCount = undefined;
          clampOpts.maxWorkgroupsX = spec.maxWorkgroupsX;
          clampOpts.maxWorkgroupsY = spec.maxWorkgroupsY;
          clampOpts.maxWorkgroupsZ = spec.maxWorkgroupsZ;
          clampOpts.maxMeshVerticesPerGroup = undefined;
          clampOpts.maxMeshTotalVertices = undefined;
          clampOpts.indirectFirstInstance = undefined;
          clampOpts.maxIndirectBatchCount = undefined;
          clampOpts.clampBatch = true;

          const exec = spec.clampExecutor || this.getIndirectClampExecutor();
          const cRes = exec.encodeClamp(
            clampTarget,
            'dispatch',
            clampOpts,
            this._scratchClampResult
          );
          if (!cRes.ok) {
            this._stats.skippedComputeDispatches += 1;
            this._recordSkip('invalid-indirect-buffer', true);
            result.skipped = true;
            result.reason = cRes.reason || 'dispatch-clamp-failed';
            return result;
          }
          if (cRes.encoded) {
            this._stats.indirectClampPasses += 1;
            this._stats.clampedIndirectDispatches += 1;
            dispatchBuf = cRes.dstBuffer;
            dispatchOff = cRes.dstOffset;
            result.dispatchClamped = true;
          }
        }
      }

      // Stage 2: Mesh compute pass (writes generated geometry + draw-indirect args).
      const ownedComputePass = !spec.computePass;
      const computePass =
        spec.computePass ||
        (commandEncoder && typeof commandEncoder.beginComputePass === 'function'
          ? commandEncoder.beginComputePass(spec.computePassDescriptor)
          : null);
      if (!computePass || typeof computePass.setPipeline !== 'function') {
        this._stats.skippedComputeDispatches += 1;
        this._recordSkip('invalid-compute-pass', false);
        result.skipped = true;
        result.reason = 'invalid-compute-pass';
        return result;
      }

      const cArgs = this._scratchMeshComputeArgs;
      cArgs.pipeline = meshPipeline;
      cArgs.bindGroups = Array.isArray(spec.meshBindGroups) ? spec.meshBindGroups : null;
      cArgs.workgroupCountX = indirectDispatch ? 0 : gx;
      cArgs.workgroupCountY = indirectDispatch ? 0 : gy;
      cArgs.workgroupCountZ = indirectDispatch ? 0 : gz;
      cArgs.indirect = indirectDispatch;
      cArgs.indirectBuffer = indirectDispatch ? dispatchBuf : null;
      cArgs.indirectOffset = indirectDispatch ? dispatchOff : 0;
      cArgs.reuseResult = true;

      const compRes = this.executeCompute(computePass, cArgs);
      if (ownedComputePass && typeof computePass.end === 'function') {
        computePass.end();
      }
      if (!compRes.executed) {
        this._recordSkip(compRes.reason || 'mesh-compute-failed', false);
        result.skipped = true;
        result.reason = compRes.reason || 'mesh-compute-failed';
        return result;
      }

      // Stage 3: Optional GPU clamp of generated draw-indirect arguments.
      const recordBytes = indexed ? 20 : 16;
      let drawBuf = spec.drawIndirectBuffer || null;
      let drawOff = spec.drawIndirectOffset || 0;
      let drawStride = spec.drawStride !== undefined ? spec.drawStride : recordBytes;

      if (spec.clampDrawIndirect !== false && canClamp) {
        const clampTarget = spec.drawClampComputePass || commandEncoder;
        if (
          clampTarget &&
          (typeof clampTarget.beginComputePass === 'function' ||
            typeof clampTarget.dispatchWorkgroups === 'function')
        ) {
          const clampOpts = this._scratchClampOptions;
          clampOpts.srcBuffer = drawBuf;
          clampOpts.srcOffset = drawOff;
          clampOpts.dstBuffer = spec.clampedDrawBuffer || null;
          clampOpts.dstOffset = spec.clampedDrawOffset || 0;
          clampOpts.countBuffer = spec.drawCountBuffer || null;
          clampOpts.countOffset = spec.drawCountOffset || 0;
          clampOpts.commandCount = drawCount;
          clampOpts.stride = drawStride;
          clampOpts.maxVertexCount = spec.maxVertexCount;
          clampOpts.maxIndexCount = spec.maxIndexCount;
          clampOpts.maxInstanceCount = spec.maxInstanceCount;
          clampOpts.maxWorkgroupsX = undefined;
          clampOpts.maxWorkgroupsY = undefined;
          clampOpts.maxWorkgroupsZ = undefined;
          clampOpts.maxMeshVerticesPerGroup = undefined;
          clampOpts.maxMeshTotalVertices = undefined;
          clampOpts.indirectFirstInstance =
            spec.indirectFirstInstance !== undefined
              ? spec.indirectFirstInstance
              : this.supportsIndirectFirstInstance();
          clampOpts.maxIndirectBatchCount = spec.maxIndirectBatchCount;
          clampOpts.clampBatch = spec.clampBatch !== false;

          const exec = spec.clampExecutor || this.getIndirectClampExecutor();
          const dClamp = exec.encodeClamp(
            clampTarget,
            indexed ? 'drawIndexed' : 'draw',
            clampOpts,
            this._scratchClampResult
          );
          if (!dClamp.ok) {
            this._recordSkip('invalid-indirect-buffer', true);
            result.skipped = true;
            result.reason = dClamp.reason || 'draw-clamp-failed';
            return result;
          }
          if (dClamp.encoded) {
            this._stats.indirectClampPasses += 1;
            if (dClamp.zeroedFirstInstance) {
              this._stats.indirectFirstInstanceClamped += dClamp.commandCount;
            }
            drawBuf = dClamp.dstBuffer;
            drawOff = dClamp.dstOffset;
            drawStride = dClamp.stride;
            result.drawClamped = true;
            result.zeroedFirstInstance = dClamp.zeroedFirstInstance;
          }
        }
      }

      // Stage 4: Render pass over generated mesh vertex/index buffers + indirect draw args.
      const ownedRenderPass = !spec.renderPass;
      const renderPass =
        spec.renderPass ||
        (commandEncoder && typeof commandEncoder.beginRenderPass === 'function'
          ? commandEncoder.beginRenderPass(spec.renderPassDescriptor)
          : null);
      if (!renderPass || typeof renderPass.setPipeline !== 'function') {
        this._recordSkip('invalid-render-pass', false);
        result.skipped = true;
        result.reason = 'invalid-render-pass';
        return result;
      }

      if (renderPipeline !== undefined && renderPipeline !== null) {
        this.setPipeline(renderPipeline);
      }
      if (Array.isArray(spec.renderBindGroups)) {
        for (let i = 0; i < spec.renderBindGroups.length; i += 1) {
          const bg = spec.renderBindGroups[i];
          if (!bg) continue;
          const groupIndex = Number.isInteger(bg.groupIndex) ? bg.groupIndex : i;
          const handle = bg.bindGroup !== undefined ? bg.bindGroup : bg;
          this.setBindGroup(groupIndex, handle, bg.dynamicOffsets || null);
        }
      }
      if (Array.isArray(spec.vertexBuffers)) {
        for (let i = 0; i < spec.vertexBuffers.length; i += 1) {
          const vb = spec.vertexBuffers[i];
          if (!vb) continue;
          const slot = Number.isInteger(vb.slot) ? vb.slot : i;
          const buf = vb.buffer !== undefined ? vb.buffer : vb;
          this.setVertexBuffer(slot, buf, vb.offset || 0, vb.size !== undefined ? vb.size : null);
        }
      }
      if (indexed && spec.indexBuffer) {
        const ib = spec.indexBuffer;
        const ibBuf = ib.buffer !== undefined ? ib.buffer : ib;
        const ibFmt = ib.format || ib.indexFormat || 'uint32';
        this.setIndexBuffer(ibBuf, ibFmt, ib.offset || 0, ib.size !== undefined ? ib.size : null);
      }

      const dArgs = this._scratchMeshDrawArgs;
      dArgs.pipeline = renderPipeline;
      dArgs.indirect = true;
      dArgs.reuseResult = true;

      let drawsIssued = 0;
      let lastFailReason = null;
      for (let i = 0, off = drawOff; i < drawCount; i += 1, off += drawStride) {
        dArgs.indirectBuffer = drawBuf;
        dArgs.indirectOffset = off;
        const dr = this._executeDrawIndirectCore(renderPass, dArgs, indexed, drawBuf, off);
        if (dr.executed) {
          drawsIssued += 1;
        } else if (lastFailReason === null) {
          lastFailReason = dr.reason;
        }
      }

      if (ownedRenderPass && typeof renderPass.end === 'function') {
        renderPass.end();
      }

      if (drawsIssued > 0) {
        this._stats.meshComputeDraws += 1;
        if (indirectDispatch) {
          this._stats.meshIndirectComputeDraws += 1;
        }
        if (result.drawClamped) {
          this._stats.clampedIndirectDraws += drawsIssued;
        }
      }

      result.ok = drawsIssued > 0;
      result.executed = drawsIssued > 0;
      result.skipped = drawsIssued === 0;
      result.drawsIssued = drawsIssued;
      result.drawIndirectBuffer = drawBuf;
      result.drawIndirectOffset = drawOff;
      result.reason = drawsIssued > 0 ? null : lastFailReason || 'mesh-draw-failed';
      return result;
    }

    /**
     * Alias for `executeMeshComputeDraw`.
     */
    dispatchMeshComputeDraw(commandEncoder, meshSpec, out) {
      return this.executeMeshComputeDraw(commandEncoder, meshSpec, out);
    }

    /**
     * Execute multi-pass / multi-subdraw instanced draws with signed i32 `baseVertex`
     * normalization and safe `firstInstance` gating when `device.features` lacks
     * `indirect-first-instance`. Preserves redundant state elision and zero per-draw GC
     * allocations on the hot path.
     *
     * @param {object} pass GPURenderPassEncoder (can be overridden per entry via `draw.pass`)
     * @param {{
     *   draws: Array<object>,
     *   commandEncoder?: object,
     *   clampComputePass?: object,
     *   firstInstancePolicy?: 'clamp' | 'zero' | 'fallback-direct' | 'strict',
     *   reuseResult?: boolean,
     *   out?: object,
     * }} spec
     * @param {object} [out]
     */
    executeMultiPassInstancedDraw(pass, spec, out) {
      const hasSpec = spec && typeof spec === 'object';
      const reuse =
        this._reuseDrawResult || (hasSpec && spec.reuseResult === true);
      const result =
        out !== undefined && out !== null && typeof out === 'object'
          ? out
          : hasSpec && spec.out && typeof spec.out === 'object'
            ? spec.out
            : reuse
              ? this._reusableMultiPassResult
              : {
                  ok: false,
                  executed: 0,
                  skipped: 0,
                  elided: 0,
                  clamped: 0,
                  fallbackDirect: 0,
                  firstInstanceClamped: 0,
                  reason: null,
                };

      result.ok = false;
      result.executed = 0;
      result.skipped = 0;
      result.elided = 0;
      result.clamped = 0;
      result.fallbackDirect = 0;
      result.firstInstanceClamped = 0;
      result.reason = null;

      if (this._destroyed) {
        this._recordSkip('executor-destroyed', false);
        result.reason = 'executor-destroyed';
        return result;
      }

      const draws = hasSpec
        ? Array.isArray(spec.draws)
          ? spec.draws
          : Array.isArray(spec.passes)
            ? spec.passes
            : null
        : null;
      if (!draws) {
        result.reason = 'invalid-draws-list';
        return result;
      }
      if (draws.length === 0) {
        result.ok = true;
        return result;
      }

      const policy =
        hasSpec && typeof spec.firstInstancePolicy === 'string'
          ? spec.firstInstancePolicy
          : 'clamp';
      const hasFirstInstanceFeature = this.supportsIndirectFirstInstance();
      const clampEncoder =
        hasSpec && (spec.clampComputePass || spec.commandEncoder)
          ? spec.clampComputePass || spec.commandEncoder
          : null;

      for (let i = 0; i < draws.length; i += 1) {
        const entry = draws[i];
        if (!entry || typeof entry !== 'object') {
          result.skipped += 1;
          continue;
        }
        const targetPass = entry.pass || pass;
        if (entry.pipeline !== undefined && entry.pipeline !== null) {
          this.setPipeline(entry.pipeline);
        }
        if (Array.isArray(entry.bindGroups)) {
          for (let b = 0; b < entry.bindGroups.length; b += 1) {
            const bg = entry.bindGroups[b];
            if (!bg) continue;
            const groupIdx = Number.isInteger(bg.groupIndex) ? bg.groupIndex : b;
            const handle = bg.bindGroup !== undefined ? bg.bindGroup : bg;
            this.setBindGroup(groupIdx, handle, bg.dynamicOffsets || null);
          }
        }
        if (Array.isArray(entry.vertexBuffers)) {
          for (let v = 0; v < entry.vertexBuffers.length; v += 1) {
            const vb = entry.vertexBuffers[v];
            if (!vb) continue;
            const slot = Number.isInteger(vb.slot) ? vb.slot : v;
            const buf = vb.buffer !== undefined ? vb.buffer : vb;
            this.setVertexBuffer(slot, buf, vb.offset || 0, vb.size !== undefined ? vb.size : null);
          }
        }
        if (entry.indexBuffer) {
          const ib = entry.indexBuffer;
          const ibBuf = ib.buffer !== undefined ? ib.buffer : ib;
          const ibFmt = ib.format || ib.indexFormat || 'uint32';
          this.setIndexBuffer(ibBuf, ibFmt, ib.offset || 0, ib.size !== undefined ? ib.size : null);
        }

        const indexed = Boolean(entry.indexed);
        const indirect = Boolean(entry.indirect === true || entry.indirectBuffer);
        const baseVertex = normalizeSignedBaseVertex(entry.baseVertex);
        const firstInstance = toNonNegativeInt(entry.firstInstance, 0);

        if (!indirect) {
          const count = indexed
            ? toNonNegativeInt(entry.indexCount, 0)
            : toNonNegativeInt(entry.vertexCount, 0);
          const instanceCount = toNonNegativeInt(
            entry.instanceCount !== undefined ? entry.instanceCount : 1,
            0
          );
          if (count <= 0 || instanceCount <= 0) {
            this._recordSkip('zero-draw-count', false);
            result.elided += 1;
            result.skipped += 1;
            continue;
          }
          const dr = this._executeDrawCore(
            targetPass,
            this._scratchReuseDrawOpts,
            indexed,
            count,
            instanceCount,
            toNonNegativeInt(entry.firstVertex, 0),
            toNonNegativeInt(entry.firstIndex, 0),
            baseVertex,
            firstInstance
          );
          if (dr.executed) {
            result.executed += 1;
          } else {
            result.skipped += 1;
            if (result.reason === null) result.reason = dr.reason;
          }
          continue;
        }

        // Indirect draw entry inside multi-pass instanced batch.
        const needsFirstInstanceGate =
          !hasFirstInstanceFeature &&
          (firstInstance > 0 || entry.clamp === true || entry.gpuWritten === true);

        if (needsFirstInstanceGate && policy === 'strict') {
          this._recordSkip('indirect-first-instance-unsupported', false);
          result.skipped += 1;
          if (result.reason === null) result.reason = 'indirect-first-instance-unsupported';
          continue;
        }

        if (
          needsFirstInstanceGate &&
          policy === 'fallback-direct' &&
          (indexed ? entry.indexCount > 0 : entry.vertexCount > 0)
        ) {
          const count = indexed
            ? toNonNegativeInt(entry.indexCount, 0)
            : toNonNegativeInt(entry.vertexCount, 0);
          const instanceCount = toNonNegativeInt(
            entry.instanceCount !== undefined ? entry.instanceCount : 1,
            0
          );
          const dr = this._executeDrawCore(
            targetPass,
            this._scratchReuseDrawOpts,
            indexed,
            count,
            instanceCount,
            toNonNegativeInt(entry.firstVertex, 0),
            toNonNegativeInt(entry.firstIndex, 0),
            baseVertex,
            firstInstance
          );
          if (dr.executed) {
            result.executed += 1;
            result.fallbackDirect += 1;
          } else {
            result.skipped += 1;
            if (result.reason === null) result.reason = dr.reason;
          }
          continue;
        }

        let drawBuf = entry.indirectBuffer || null;
        let drawOff = entry.indirectOffset || 0;
        const entryClampTarget =
          entry.clampComputePass || entry.commandEncoder || clampEncoder;

        if ((entry.clamp === true || needsFirstInstanceGate) && entryClampTarget) {
          const clampRes = this.clampIndirectArgs(
            entryClampTarget,
            indexed ? 'drawIndexed' : 'draw',
            drawBuf,
            drawOff,
            entry,
            this._scratchClampResult
          );
          if (!clampRes.ok) {
            this._recordSkip('invalid-indirect-buffer', true);
            result.skipped += 1;
            if (result.reason === null) result.reason = clampRes.reason || 'indirect-clamp-failed';
            continue;
          }
          if (!clampRes.encoded) {
            result.elided += 1;
            result.skipped += 1;
            continue;
          }
          drawBuf = clampRes.dstBuffer;
          drawOff = clampRes.dstOffset;
          result.clamped += 1;
          if (clampRes.zeroedFirstInstance) {
            result.firstInstanceClamped += 1;
          }
        }

        const dArgs = this._scratchMeshDrawArgs;
        dArgs.pipeline = entry.pipeline !== undefined ? entry.pipeline : this._activePipeline;
        dArgs.indirect = true;
        dArgs.indirectBuffer = drawBuf;
        dArgs.indirectOffset = drawOff;
        dArgs.reuseResult = true;
        const dr = this._executeDrawIndirectCore(targetPass, dArgs, indexed, drawBuf, drawOff);
        if (dr.executed) {
          result.executed += 1;
          if (entry.clamp === true || (needsFirstInstanceGate && entryClampTarget)) {
            this._stats.clampedIndirectDraws += 1;
          }
        } else {
          result.skipped += 1;
          if (result.reason === null) result.reason = dr.reason;
        }
      }

      if (result.executed > 0) {
        this._stats.multiPassInstancedDraws += 1;
        this._stats.multiPassInstancedSubDraws += result.executed;
      }
      result.ok = result.executed > 0 || (result.skipped === 0 && result.elided > 0);
      return result;
    }

    /**
     * Execute a non-indexed draw (`DrawInstanced`) with automatic fallback profile routing
     * and hardened bind-group / vertex-buffer guards.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ vertexCount, instanceCount?, firstVertex?, firstInstance?, ...overrides }`
     */
    executeDraw(pass, drawArgs = {}) {
      return this._executeDrawInternal(pass, drawArgs, false);
    }

    /**
     * Execute an indexed draw (`DrawIndexedInstanced`) with automatic fallback profile routing
     * and hardened index / bind-group / vertex-buffer guards.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?, ...overrides }`
     */
    executeDrawIndexed(pass, drawArgs = {}) {
      return this._executeDrawInternal(pass, drawArgs, true);
    }

    /**
     * Execute an indirect non-indexed draw (`drawIndirect`) with redundant-pipeline elision
     * and indirect buffer validation.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indirectBuffer, indirectOffset?, ...overrides }`
     */
    executeDrawIndirect(pass, drawArgs = {}) {
      if (
        drawArgs &&
        drawArgs.clamp === true &&
        (drawArgs.commandEncoder || drawArgs.clampComputePass)
      ) {
        return this.executeClampedDrawIndirect(drawArgs.commandEncoder || null, pass, drawArgs);
      }
      return this._executeDrawIndirectInternal(pass, drawArgs, false);
    }

    /**
     * Execute an indirect indexed draw (`drawIndexedIndirect`) with redundant-pipeline elision
     * and index / indirect buffer validation.
     *
     * @param {object} pass GPURenderPassEncoder
     * @param {object} drawArgs `{ indirectBuffer, indirectOffset?, ...overrides }`
     */
    executeDrawIndexedIndirect(pass, drawArgs = {}) {
      if (
        drawArgs &&
        drawArgs.clamp === true &&
        (drawArgs.commandEncoder || drawArgs.clampComputePass)
      ) {
        return this.executeClampedDrawIndexedIndirect(
          drawArgs.commandEncoder || null,
          pass,
          drawArgs
        );
      }
      return this._executeDrawIndirectInternal(pass, drawArgs, true);
    }

    /**
     * Positional convenience wrapper for `executeDraw` (zero-alloc argument path).
     */
    draw(pass, vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
      return this._executeDrawCore(
        pass,
        null,
        false,
        toNonNegativeInt(vertexCount, 0),
        toNonNegativeInt(instanceCount, 0),
        toNonNegativeInt(firstVertex, 0),
        0,
        0,
        toNonNegativeInt(firstInstance, 0)
      );
    }

    /**
     * Positional convenience wrapper for `executeDrawIndexed` (zero-alloc argument path).
     */
    drawIndexed(pass, indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
      return this._executeDrawCore(
        pass,
        null,
        true,
        toNonNegativeInt(indexCount, 0),
        toNonNegativeInt(instanceCount, 0),
        0,
        toNonNegativeInt(firstIndex, 0),
        Number.isInteger(baseVertex) ? baseVertex : 0,
        toNonNegativeInt(firstInstance, 0)
      );
    }

    /**
     * Positional convenience wrapper for `executeDrawIndirect` (zero-alloc argument path).
     */
    drawIndirect(pass, indirectBuffer, indirectOffset = 0) {
      return this._executeDrawIndirectCore(
        pass,
        null,
        false,
        indirectBuffer !== undefined ? indirectBuffer : null,
        indirectOffset !== undefined ? indirectOffset : 0
      );
    }

    /**
     * Positional convenience wrapper for `executeDrawIndexedIndirect` (zero-alloc argument path).
     */
    drawIndexedIndirect(pass, indirectBuffer, indirectOffset = 0) {
      return this._executeDrawIndirectCore(
        pass,
        null,
        true,
        indirectBuffer !== undefined ? indirectBuffer : null,
        indirectOffset !== undefined ? indirectOffset : 0
      );
    }

    _executeDrawIndirectInternal(pass, drawArgs, indexed) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : null;
      const indirectBuffer =
        args !== null && args.indirectBuffer !== undefined ? args.indirectBuffer : null;
      const indirectOffset =
        args !== null && args.indirectOffset !== undefined ? args.indirectOffset : 0;
      return this._executeDrawIndirectCore(pass, args, indexed, indirectBuffer, indirectOffset);
    }

    _executeDrawIndirectCore(pass, args, indexed, indirectBuffer, indirectOffset) {
      const safeOffset =
        Number.isInteger(indirectOffset) && indirectOffset >= 0 ? indirectOffset : 0;
      const result = this._acquireDrawResult(
        indexed,
        true,
        0,
        0,
        safeOffset,
        args !== null ? args.reuseResult : false
      );

      if (this._destroyed) {
        this._recordSkip('executor-destroyed', false);
        result.skipped = true;
        result.reason = 'executor-destroyed';
        return result;
      }

      const drawMethod = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
      if (
        !pass ||
        typeof pass !== 'object' ||
        typeof pass.setPipeline !== 'function' ||
        typeof pass[drawMethod] !== 'function'
      ) {
        this._recordSkip('invalid-render-pass', false);
        result.skipped = true;
        result.reason = 'invalid-render-pass';
        return result;
      }

      const indStatus = validateIndirectBufferStatus(indirectBuffer, indirectOffset, indexed);
      if (indStatus !== INDIRECT_STATUS_OK) {
        this._recordSkip('invalid-indirect-buffer', indStatus === INDIRECT_STATUS_INVALID_RANGE);
        result.skipped = true;
        result.reason = 'invalid-indirect-buffer';
        return result;
      }

      const activePipe =
        args !== null && args.pipeline !== undefined ? args.pipeline : this._activePipeline;
      const isFallback = this._needsFallbackRoutingFast(args, activePipe);
      result.isFallback = isFallback;

      if (!isFallback) {
        const realPipeline =
          activePipe && activePipe.pipeline ? activePipe.pipeline : activePipe;

        try {
          const bound = this._bindPipelineOnPass(pass, realPipeline);
          result.pipelineElided = !bound;
          this._bindPassDynamicState(pass, args, result);
          this._bindNormalResources(pass, result);
          if (indexed) {
            const ibStatus = this._validateAndBindIndexBuffer(pass, 1, 0);
            if (ibStatus === IB_STATUS_INVALID || ibStatus === IB_STATUS_INVALID_RANGE) {
              this._recordSkip('invalid-index-buffer', ibStatus === IB_STATUS_INVALID_RANGE);
              result.skipped = true;
              result.reason = 'invalid-index-buffer';
              return result;
            }
            result.indexBufferElided = ibStatus === IB_STATUS_ELIDED;
            pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
          } else {
            pass.drawIndirect(indirectBuffer, indirectOffset);
          }
          this._stats.totalDraws += 1;
          this._stats.normalDraws += 1;
          this._stats.indirectDraws += 1;
          this._recordExecutedDrawVisibility(args, activePipe, false);
          this._recordExecutedDrawCategory(args, activePipe, 0, null);
          result.executed = true;
          return result;
        } catch (err) {
          this._stats.passErrors += 1;
          this._recordSkip('pass-error', false);
          this._warn('Normal indirect draw execution failed', {
            error: String(err && err.message ? err.message : err),
          });
          result.skipped = true;
          result.reason = 'pass-error';
          return result;
        }
      }

      if (!this._enableShaderFallback && !this._shaderFallbackRouter) {
        this._recordSkip('no-pipeline', false);
        result.skipped = true;
        result.reason = 'no-pipeline';
        return result;
      }

      const drawState = this.getDrawState({
        ...(args !== null ? args : {}),
        indexed,
        indirect: true,
        vertexCount: args !== null && args.vertexCount !== undefined ? args.vertexCount : 0,
        indexCount: 0,
        instanceCount: 1,
      });

      const resolved = this.resolveFallbackShaders(drawState);
      if (!resolved) {
        this._recordSkip('shader-resolve-failed', false);
        result.skipped = true;
        result.reason = 'shader-resolve-failed';
        return result;
      }

      result.profile = resolved.selectedProfile;
      result.effectiveProfile = resolved.effectiveProfile;

      const fallbackPipeline = this._getOrCreateFallbackPipeline(resolved, drawState);
      if (!fallbackPipeline) {
        this._recordSkip('pipeline-creation-failed', false);
        result.skipped = true;
        result.reason = 'pipeline-creation-failed';
        return result;
      }

      try {
        const bound = this._bindPipelineOnPass(pass, fallbackPipeline);
        result.pipelineElided = !bound;
      } catch (err) {
        this._stats.passErrors += 1;
        this._recordSkip('pass-error', false);
        this._warn('pass.setPipeline failed on fallback indirect draw', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        result.skipped = true;
        result.reason = 'pass-error';
        return result;
      }

      if (resolved.effectiveProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
        const bgRes = this._bindFallbackGroupForCbv(pass, fallbackPipeline, drawState);
        result.substitutedCbv = bgRes.substitutedCbv;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-cbv-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-cbv-bind-failed';
          return result;
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
        const bgRes = this._bindFallbackGroupForTexturedQuad(pass, fallbackPipeline, drawState);
        result.substitutedTexture = bgRes.substitutedTexture;
        result.substitutedSampler = bgRes.substitutedSampler;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-texture-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-texture-bind-failed';
          return result;
        }

        const vbRes = this._bindFallbackVertexBuffer24B(pass, drawState, 1, 0);
        result.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (vbRes.vertexBufferElided) result.vertexBuffersElided += 1;
        if (!vbRes.ok) {
          this._recordSkip('fallback-vb-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-vb-bind-failed';
          return result;
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B) {
        const bgRes = this._bindFallbackGroupForSplashRect(pass, fallbackPipeline, drawState);
        result.substitutedCbv = bgRes.substitutedCbv;
        result.substitutedTexture = bgRes.substitutedTexture;
        result.substitutedSampler = bgRes.substitutedSampler;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-splash-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-splash-bind-failed';
          return result;
        }

        const vbRes = this._bindFallbackVertexBufferSplashRect24B(pass, drawState, 1, 0);
        result.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (vbRes.vertexBufferElided) result.vertexBuffersElided += 1;
        if (!vbRes.ok) {
          this._recordSkip('fallback-vb-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-vb-bind-failed';
          return result;
        }
      }

      try {
        this._bindPassDynamicState(pass, args, result);
        if (indexed) {
          const ibStatus = this._validateAndBindIndexBuffer(pass, 1, 0);
          if (ibStatus === IB_STATUS_INVALID || ibStatus === IB_STATUS_INVALID_RANGE) {
            this._recordSkip('invalid-index-buffer', ibStatus === IB_STATUS_INVALID_RANGE);
            result.skipped = true;
            result.reason = 'invalid-index-buffer';
            return result;
          }
          result.indexBufferElided = ibStatus === IB_STATUS_ELIDED;
          pass.drawIndexedIndirect(indirectBuffer, indirectOffset);
        } else {
          pass.drawIndirect(indirectBuffer, indirectOffset);
        }
      } catch (err) {
        this._stats.passErrors += 1;
        this._recordSkip('pass-error', false);
        this._warn('Fallback indirect draw submission failed', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        result.skipped = true;
        result.reason = 'pass-error';
        return result;
      }

      this._stats.totalDraws += 1;
      this._stats.fallbackDraws += 1;
      this._stats.indirectDraws += 1;
      const countedProfile = Object.prototype.hasOwnProperty.call(
        this._stats.fallbackProfileCounts,
        resolved.selectedProfile
      )
        ? resolved.selectedProfile
        : FALLBACK_PROFILE.DEFAULT;
      this._stats.fallbackProfileCounts[countedProfile] += 1;
      this._recordExecutedDrawVisibility(args, activePipe, true);
      this._recordExecutedDrawCategory(args, activePipe, 0, resolved.effectiveProfile);

      result.executed = true;
      return result;
    }

    _executeDrawInternal(pass, drawArgs, indexed) {
      const args = drawArgs && typeof drawArgs === 'object' ? drawArgs : null;
      if (args !== null && args.indirect === true) {
        return this._executeDrawIndirectInternal(pass, args, indexed);
      }
      const count =
        args === null
          ? 0
          : indexed
            ? toNonNegativeInt(args.indexCount, 0)
            : toNonNegativeInt(args.vertexCount, 0);
      const instanceCount =
        args === null
          ? 1
          : toNonNegativeInt(args.instanceCount !== undefined ? args.instanceCount : 1, 0);
      const firstVertex = args === null ? 0 : toNonNegativeInt(args.firstVertex, 0);
      const firstIndex = args === null ? 0 : toNonNegativeInt(args.firstIndex, 0);
      const baseVertex = args !== null && Number.isInteger(args.baseVertex) ? args.baseVertex : 0;
      const firstInstance = args === null ? 0 : toNonNegativeInt(args.firstInstance, 0);

      return this._executeDrawCore(
        pass,
        args,
        indexed,
        count,
        instanceCount,
        firstVertex,
        firstIndex,
        baseVertex,
        firstInstance
      );
    }

    _executeDrawCore(
      pass,
      args,
      indexed,
      count,
      instanceCount,
      firstVertex,
      firstIndex,
      baseVertex,
      firstInstance
    ) {
      const result = this._acquireDrawResult(
        indexed,
        false,
        count,
        instanceCount,
        0,
        args !== null ? args.reuseResult : false
      );

      if (this._destroyed) {
        this._recordSkip('executor-destroyed', false);
        result.skipped = true;
        result.reason = 'executor-destroyed';
        return result;
      }

      if (
        !pass ||
        typeof pass !== 'object' ||
        typeof pass.setPipeline !== 'function' ||
        typeof (indexed ? pass.drawIndexed : pass.draw) !== 'function'
      ) {
        this._recordSkip('invalid-render-pass', false);
        result.skipped = true;
        result.reason = 'invalid-render-pass';
        return result;
      }

      if (count <= 0 || instanceCount <= 0) {
        this._recordSkip('zero-draw-count', false);
        result.skipped = true;
        result.reason = 'zero-draw-count';
        return result;
      }

      const activePipe =
        args !== null && args.pipeline !== undefined ? args.pipeline : this._activePipeline;
      const isFallback = this._needsFallbackRoutingFast(args, activePipe);
      result.isFallback = isFallback;

      if (!isFallback) {
        const realPipeline =
          activePipe && activePipe.pipeline ? activePipe.pipeline : activePipe;

        try {
          const bound = this._bindPipelineOnPass(pass, realPipeline);
          result.pipelineElided = !bound;
          this._bindPassDynamicState(pass, args, result);
          this._bindNormalResources(pass, result);
          if (indexed) {
            const ibStatus = this._validateAndBindIndexBuffer(pass, count, firstIndex);
            if (ibStatus === IB_STATUS_INVALID || ibStatus === IB_STATUS_INVALID_RANGE) {
              this._recordSkip('invalid-index-buffer', ibStatus === IB_STATUS_INVALID_RANGE);
              result.skipped = true;
              result.reason = 'invalid-index-buffer';
              return result;
            }
            result.indexBufferElided = ibStatus === IB_STATUS_ELIDED;
            pass.drawIndexed(count, instanceCount, firstIndex, baseVertex, firstInstance);
            this._stats.indexedDraws += 1;
          } else {
            pass.draw(count, instanceCount, firstVertex, firstInstance);
            this._stats.nonIndexedDraws += 1;
          }
          this._stats.totalDraws += 1;
          this._stats.normalDraws += 1;
          this._recordExecutedDrawVisibility(args, activePipe, false);
          this._recordExecutedDrawCategory(args, activePipe, count, null);
          result.executed = true;
          return result;
        } catch (err) {
          this._stats.passErrors += 1;
          this._recordSkip('pass-error', false);
          this._warn('Normal draw execution failed', {
            error: String(err && err.message ? err.message : err),
          });
          result.skipped = true;
          result.reason = 'pass-error';
          return result;
        }
      }

      if (!this._enableShaderFallback && !this._shaderFallbackRouter) {
        this._recordSkip('no-pipeline', false);
        result.skipped = true;
        result.reason = 'no-pipeline';
        return result;
      }

      const drawState = this.getDrawState({
        ...(args !== null ? args : {}),
        indexed,
        vertexCount: indexed ? (args !== null && args.vertexCount ? args.vertexCount : 0) : count,
        indexCount: indexed ? count : 0,
        instanceCount,
        firstVertex,
        firstIndex,
        baseVertex,
        firstInstance,
      });

      // Fallback draw path: select G34 profile, resolve shaders, create/cache pipeline,
      // and harden bind-group & vertex-buffer bindings.
      const resolved = this.resolveFallbackShaders(drawState);
      if (!resolved) {
        this._recordSkip('shader-resolve-failed', false);
        result.skipped = true;
        result.reason = 'shader-resolve-failed';
        return result;
      }

      result.profile = resolved.selectedProfile;
      result.effectiveProfile = resolved.effectiveProfile;

      const fallbackPipeline = this._getOrCreateFallbackPipeline(resolved, drawState);
      if (!fallbackPipeline) {
        this._recordSkip('pipeline-creation-failed', false);
        result.skipped = true;
        result.reason = 'pipeline-creation-failed';
        return result;
      }

      try {
        const bound = this._bindPipelineOnPass(pass, fallbackPipeline);
        result.pipelineElided = !bound;
      } catch (err) {
        this._stats.passErrors += 1;
        this._recordSkip('pass-error', false);
        this._warn('pass.setPipeline failed on fallback draw', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        result.skipped = true;
        result.reason = 'pass-error';
        return result;
      }

      let effectiveCount = count;
      let effectiveFirstVertex = firstVertex;

      if (resolved.effectiveProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
        const bgRes = this._bindFallbackGroupForCbv(pass, fallbackPipeline, drawState);
        result.substitutedCbv = bgRes.substitutedCbv;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-cbv-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-cbv-bind-failed';
          return result;
        }
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
        const bgRes = this._bindFallbackGroupForTexturedQuad(pass, fallbackPipeline, drawState);
        result.substitutedTexture = bgRes.substitutedTexture;
        result.substitutedSampler = bgRes.substitutedSampler;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-texture-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-texture-bind-failed';
          return result;
        }

        const vbRes = this._bindFallbackVertexBuffer24B(pass, drawState, count, firstVertex);
        result.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (vbRes.vertexBufferElided) result.vertexBuffersElided += 1;
        if (!vbRes.ok) {
          this._recordSkip('fallback-vb-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-vb-bind-failed';
          return result;
        }
        effectiveCount = vbRes.effectiveVertexCount;
        effectiveFirstVertex = vbRes.effectiveFirstVertex;
      } else if (resolved.effectiveProfile === FALLBACK_PROFILE.SPLASH_RECT_24B) {
        const bgRes = this._bindFallbackGroupForSplashRect(pass, fallbackPipeline, drawState);
        result.substitutedCbv = bgRes.substitutedCbv;
        result.substitutedTexture = bgRes.substitutedTexture;
        result.substitutedSampler = bgRes.substitutedSampler;
        result.bindGroupCacheHit = bgRes.cacheHit;
        if (bgRes.bindGroupElided) result.bindGroupsElided += 1;
        if (!bgRes.ok) {
          this._recordSkip('fallback-splash-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-splash-bind-failed';
          return result;
        }

        const vbRes = this._bindFallbackVertexBufferSplashRect24B(
          pass,
          drawState,
          count,
          firstVertex
        );
        result.substitutedVertexBuffer = vbRes.substitutedVertexBuffer;
        if (vbRes.vertexBufferElided) result.vertexBuffersElided += 1;
        if (!vbRes.ok) {
          this._recordSkip('fallback-vb-bind-failed', false);
          result.skipped = true;
          result.reason = 'fallback-vb-bind-failed';
          return result;
        }
        effectiveCount = vbRes.effectiveVertexCount;
        effectiveFirstVertex = vbRes.effectiveFirstVertex;
      }

      try {
        this._bindPassDynamicState(pass, args, result);
        if (indexed) {
          const ibStatus = this._validateAndBindIndexBuffer(pass, effectiveCount, firstIndex);
          if (ibStatus === IB_STATUS_INVALID || ibStatus === IB_STATUS_INVALID_RANGE) {
            this._recordSkip('invalid-index-buffer', ibStatus === IB_STATUS_INVALID_RANGE);
            result.skipped = true;
            result.reason = 'invalid-index-buffer';
            return result;
          }
          result.indexBufferElided = ibStatus === IB_STATUS_ELIDED;
          pass.drawIndexed(effectiveCount, instanceCount, firstIndex, baseVertex, firstInstance);
          this._stats.indexedDraws += 1;
        } else {
          pass.draw(effectiveCount, instanceCount, effectiveFirstVertex, firstInstance);
          this._stats.nonIndexedDraws += 1;
        }
      } catch (err) {
        this._stats.passErrors += 1;
        this._recordSkip('pass-error', false);
        this._warn('Fallback draw submission failed', {
          profile: resolved.effectiveProfile,
          error: String(err && err.message ? err.message : err),
        });
        result.skipped = true;
        result.reason = 'pass-error';
        return result;
      }

      this._stats.totalDraws += 1;
      this._stats.fallbackDraws += 1;
      const countedProfile = Object.prototype.hasOwnProperty.call(
        this._stats.fallbackProfileCounts,
        resolved.selectedProfile
      )
        ? resolved.selectedProfile
        : FALLBACK_PROFILE.DEFAULT;
      this._stats.fallbackProfileCounts[countedProfile] += 1;
      this._recordExecutedDrawVisibility(args, activePipe, true);
      this._recordExecutedDrawCategory(args, activePipe, effectiveCount, resolved.effectiveProfile);

      result.executed = true;
      result.vertexCount = indexed ? 0 : effectiveCount;
      result.indexCount = indexed ? effectiveCount : 0;
      return result;
    }

    /**
     * Return a fresh JSON-serializable snapshot of draw executor telemetry,
     * including `fallbackProfileCounts`, per-kind draw counters, redundant-state elision,
     * bind-group cache metrics, UI/glyph/video draw accounting, and per-reason skip counts.
     */
    getStats() {
      const skippedByReason = {};
      const reasonKeys = Object.keys(this._stats.skippedByReason);
      for (let i = 0; i < reasonKeys.length; i += 1) {
        const k = reasonKeys[i];
        skippedByReason[k] = this._stats.skippedByReason[k];
      }
      return {
        totalDraws: this._stats.totalDraws,
        totalDrawsExecuted: this._stats.totalDrawsExecuted,
        normalDraws: this._stats.normalDraws,
        fallbackDraws: this._stats.fallbackDraws,
        hasNonClearDraw: this._frameHasNonClearDraw,
        realPackDrawsThisFrame: this._frameRealPackDraws,
        fallbackDrawsThisFrame: this._frameFallbackDraws,
        drawsThisFrame: this._frameDraws,
        indexedDraws: this._stats.indexedDraws,
        nonIndexedDraws: this._stats.nonIndexedDraws,
        indirectDraws: this._stats.indirectDraws,
        skippedDraws: this._stats.skippedDraws,
        zeroCountDrawsSkipped: this._stats.zeroCountDrawsSkipped,
        skippedNoPipeline: this._stats.skippedNoPipeline,
        skippedNoPass: this._stats.skippedNoPass,
        skippedInvalidRange: this._stats.skippedInvalidRange,
        skippedInvalidIndexBuffer: this._stats.skippedInvalidIndexBuffer,
        skippedInvalidIndirectBuffer: this._stats.skippedInvalidIndirectBuffer,
        skippedBindFailed: this._stats.skippedBindFailed,
        skippedPassError: this._stats.skippedPassError,
        skippedDestroyed: this._stats.skippedDestroyed,
        skippedByReason,
        redundantPipelineSkips: this._stats.redundantPipelineSkips,
        redundantBindGroupSkips: this._stats.redundantBindGroupSkips,
        redundantVertexBufferSkips: this._stats.redundantVertexBufferSkips,
        redundantIndexBufferSkips: this._stats.redundantIndexBufferSkips,
        elidedPipelineSets: this._stats.elidedPipelineSets,
        elidedBindGroupSets: this._stats.elidedBindGroupSets,
        elidedVertexBufferSets: this._stats.elidedVertexBufferSets,
        elidedIndexBufferSets: this._stats.elidedIndexBufferSets,
        elidedViewportScissorSets: this._stats.elidedViewportScissorSets,
        elidedBlendConstantSets: this._stats.elidedBlendConstantSets,
        elidedStencilReferenceSets: this._stats.elidedStencilReferenceSets,
        totalElidedStateCalls: this._stats.totalElidedStateCalls,
        stateElisionHitRate: this._computeStateElisionHitRate(),
        bindGroupCacheHits: this._stats.bindGroupCacheHits,
        bindGroupCacheMisses: this._stats.bindGroupCacheMisses,
        bindGroupCacheEvictions: this._stats.bindGroupCacheEvictions,
        bindGroupCacheSize: this._bgCacheSize,
        textureViewCacheHits: this._stats.textureViewCacheHits,
        textureViewCacheMisses: this._stats.textureViewCacheMisses,
        textureViewCacheEvictions: this._stats.textureViewCacheEvictions,
        computeDispatches: this._stats.computeDispatches,
        skippedComputeDispatches: this._stats.skippedComputeDispatches,
        alphaBlendedDraws: this._stats.alphaBlendedDraws,
        uiQuadDraws: this._stats.uiQuadDraws,
        fontGlyphDraws: this._stats.fontGlyphDraws,
        videoDraws: this._stats.videoDraws,
        fallbackProfileCounts: {
          [FALLBACK_PROFILE.DEFAULT]: this._stats.fallbackProfileCounts[FALLBACK_PROFILE.DEFAULT],
          [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV],
          [FALLBACK_PROFILE.TEXTURED_QUAD_24B]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.TEXTURED_QUAD_24B],
          [FALLBACK_PROFILE.SPLASH_RECT_24B]:
            this._stats.fallbackProfileCounts[FALLBACK_PROFILE.SPLASH_RECT_24B],
        },
        fallbackPipelineCacheHits: this._stats.fallbackPipelineCacheHits,
        fallbackPipelineCacheMisses: this._stats.fallbackPipelineCacheMisses,
        bindGroupGuards: this._stats.bindGroupGuards,
        invalidCbvSubstitutions: this._stats.invalidCbvSubstitutions,
        invalidTextureSubstitutions: this._stats.invalidTextureSubstitutions,
        invalidSamplerSubstitutions: this._stats.invalidSamplerSubstitutions,
        skippedBindGroupBindings: this._stats.skippedBindGroupBindings,
        invalidVertexBufferGuards: this._stats.invalidVertexBufferGuards,
        vertexBufferSubstitutions: this._stats.vertexBufferSubstitutions,
        skippedVertexBufferBindings: this._stats.skippedVertexBufferBindings,
        pipelineErrors: this._stats.pipelineErrors,
        bindGroupErrors: this._stats.bindGroupErrors,
        passErrors: this._stats.passErrors,
        legacyRouterFallbacks: this._stats.legacyRouterFallbacks,
        clampedIndirectDraws: this._stats.clampedIndirectDraws,
        clampedIndirectDispatches: this._stats.clampedIndirectDispatches,
        indirectClampPasses: this._stats.indirectClampPasses,
        indirectFirstInstanceClamped: this._stats.indirectFirstInstanceClamped,
        meshComputeDraws: this._stats.meshComputeDraws,
        meshIndirectComputeDraws: this._stats.meshIndirectComputeDraws,
        meshZeroCountElided: this._stats.meshZeroCountElided,
        multiPassInstancedDraws: this._stats.multiPassInstancedDraws,
        multiPassInstancedSubDraws: this._stats.multiPassInstancedSubDraws,
      };
    }

    /**
     * Reset all draw and fallback telemetry counters.
     */
    resetStats() {
      this._stats = createZeroedStats();
      this._totalStateCalls = 0;
      this.beginFrame();
    }

    _destroySyntheticResources() {
      const destroySafe = (res) => {
        if (res && typeof res.destroy === 'function') {
          try {
            res.destroy();
          } catch (_err) {
            // Ignore cleanup errors on already-lost devices.
          }
        }
      };
      destroySafe(this._fallbackCbvBuffer);
      destroySafe(this._fallbackTexture);
      destroySafe(this._fallbackVertexBuffer24B);
      destroySafe(this._fallbackSplashVertexBuffer24B);
      this._fallbackCbvBuffer = null;
      this._fallbackTexture = null;
      this._fallbackTextureView = null;
      this._fallbackSampler = null;
      this._fallbackVertexBuffer24B = null;
      this._fallbackSplashVertexBuffer24B = null;
    }

    /**
     * Release synthetic GPU resources and clear caches.
     */
    destroy() {
      this._destroyed = true;
      this._destroySyntheticResources();
      if (this._ownsIndirectClampExecutor && this._indirectClampExecutor) {
        try {
          this._indirectClampExecutor.destroy();
        } catch (_err) {
          // Ignore cleanup errors on already-lost devices.
        }
        this._indirectClampExecutor = null;
        this._ownsIndirectClampExecutor = false;
      }
      this._fallbackPipelineCache.clear();
      this._fallbackShaderModuleCache.clear();
      this._clearBgCacheInternal();
      this._wgslHashCache.clear();
      this._textureViewCache = new WeakMap();
      this._bindGroupLayoutCache = new WeakMap();
      this.resetState();
    }
  }

  DrawExecutor.selectFallbackShaderProfile = selectFallbackShaderProfile;
  DrawExecutor.expandRectListVertices24B = expandRectListVertices24B;

  const exportedApi = Object.freeze({
    FALLBACK_PROFILE,
    TEXTURED_QUAD_VERTEX_STRIDE,
    FRAME1_CBV_MIN_BYTES,
    FALLBACK_VERTEX_LAYOUT_24B,
    FRAME1_SPLASH_VERTEX_LAYOUT_24B,
    expandRectListVertices24B,
    selectFallbackShaderProfile,
    DrawExecutor,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6DrawExecutor = exportedApi;
  } else if (root) {
    root.PlayGta6DrawExecutor = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
