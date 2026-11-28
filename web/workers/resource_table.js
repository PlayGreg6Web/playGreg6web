'use strict';

// Hardened WebGPU ResourceTable for CreateTexture2 (ABI v2 opcode 35),
// BC1..BC7 feature fallback & 4x4 dimension alignment, and WebGPU buffer
// size/write 4-byte alignment padding.
// Zero external dependencies; UMD/CommonJS/ESM compatible for classic Web Workers
// (importScripts -> globalThis.PlayGta6ResourceTable), browsers, and Node.js.

(function (globalScope) {
  const CREATE_TEXTURE2_OPCODE = 35;
  const CREATE_TEXTURE2_PAYLOAD_BYTES = 40;
  const MAX_TEXTURE_VIEW_FORMATS = 8;

  const TEXTURE_COMPRESSION_BC_FEATURE = 'texture-compression-bc';
  const BC_FALLBACK_FORMAT = 'rgba8unorm';
  const BC_FALLBACK_SRGB_FORMAT = 'rgba8unorm-srgb';

  const BUFFER_ALIGNMENT_BYTES = 4;
  const UNIFORM_BUFFER_MIN_BYTES = 16;
  const DEFAULT_MAX_WRITE_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB cap per queue.writeBuffer (W352)
  const DEFAULT_FAST_SLOT_CAPACITY = 1024;
  const DEFAULT_MAX_BIND_GROUPS = 4096;
  const DEFAULT_FAST_BIND_GROUP_BUCKETS = 512;
  const FAST_BIND_GROUP_PROBE_WINDOW = 8;
  const MAX_FAST_BIND_GROUP_ENTRIES = 64;
  const MAX_GENERATION_TOMBSTONES = 65536;

  // Dual-u32 FNV-1a + splitmix32 constants for zero-alloc bind-group key hashing (W320).
  const FNV1A_32_OFFSET = 0x811c9dc5;
  const FNV1A_32_PRIME = 0x01000193;
  const FNV1A_HI_OFFSET = 0x9e3779b9;
  const SPLITMIX_C1 = 0x85ebca6b;
  const SPLITMIX_C2 = 0xc2b2ae35;
  const TWO_POW_32 = 4294967296;

  const BG_KIND_BUFFER = 1;
  const BG_KIND_TEXTURE = 2;
  const BG_KIND_TEXTURE_VIEW = 3;
  const BG_KIND_SAMPLER = 4;
  const BG_KIND_SLOT = 5;
  const BG_KIND_EXT_BUFFER = 6;
  const BG_KIND_EXT_OBJECT = 7;

  function splitmix32(x) {
    let z = (x + 0x9e3779b9) >>> 0;
    z = Math.imul(z ^ (z >>> 16), SPLITMIX_C1) >>> 0;
    z = Math.imul(z ^ (z >>> 13), SPLITMIX_C2) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  }

  function nextPowerOfTwo(n) {
    let p = 1;
    while (p < n && p < 0x40000000) {
      p <<= 1;
    }
    return p;
  }

  /**
   * Compute a zero-allocation deterministic dual-u32 (64-bit) FNV-1a + splitmix32
   * hash over `(layoutId, entryCount, bindingSlots[], resourceHandles[], generations[])`
   * with optional extra typed arrays `(kinds[], offsets[], sizes[])`.
   *
   * When `outPair` (e.g. a pre-allocated `Uint32Array(2)`) is provided, writes
   * the full 64-bit `(lo, hi)` state into `outPair[0]` and `outPair[1]`.
   * Returns a non-negative 53-bit safe integer combining both 32-bit halves with
   * zero heap allocations.
   *
   * @param {number | string} layoutId
   * @param {number} entryCount
   * @param {ArrayLike<number> | null | undefined} bindingSlots
   * @param {ArrayLike<number> | null | undefined} resourceHandles
   * @param {ArrayLike<number> | null | undefined} generations
   * @param {Uint32Array | Array<number> | null | undefined} [outPair]
   * @param {ArrayLike<number> | null | undefined} [kinds]
   * @param {ArrayLike<number> | null | undefined} [offsets]
   * @param {ArrayLike<number> | null | undefined} [sizes]
   * @returns {number}
   */
  function computeBindGroupHashFast(
    layoutId,
    entryCount,
    bindingSlots,
    resourceHandles,
    generations,
    outPair,
    kinds,
    offsets,
    sizes
  ) {
    let h1 = FNV1A_32_OFFSET;
    let h2 = FNV1A_HI_OFFSET;

    if (typeof layoutId === 'number') {
      const lo = layoutId >>> 0;
      const hi = layoutId > 0xffffffff ? Math.floor(layoutId / TWO_POW_32) >>> 0 : 0;
      h1 = Math.imul(h1 ^ lo, FNV1A_32_PRIME) >>> 0;
      h2 = Math.imul(h2 ^ (hi ^ splitmix32(lo)), FNV1A_32_PRIME) >>> 0;
    } else if (typeof layoutId === 'string') {
      const strLen = layoutId.length;
      for (let i = 0; i < strLen; i += 1) {
        const ch = layoutId.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, FNV1A_32_PRIME) >>> 0;
        h2 = Math.imul(h2 ^ splitmix32((ch + i) >>> 0), FNV1A_32_PRIME) >>> 0;
      }
    } else {
      h1 = Math.imul(h1, FNV1A_32_PRIME) >>> 0;
      h2 = Math.imul(h2 ^ 0x27d4eb2d, FNV1A_32_PRIME) >>> 0;
    }

    const count = entryCount > 0 ? entryCount >>> 0 : 0;
    h1 = Math.imul(h1 ^ count, FNV1A_32_PRIME) >>> 0;
    h2 = Math.imul(h2 ^ splitmix32(count), FNV1A_32_PRIME) >>> 0;

    const hasSlots = bindingSlots !== undefined && bindingSlots !== null;
    const hasHandles = resourceHandles !== undefined && resourceHandles !== null;
    const hasGens = generations !== undefined && generations !== null;
    const hasKinds = kinds !== undefined && kinds !== null;
    const hasOffsets = offsets !== undefined && offsets !== null;
    const hasSizes = sizes !== undefined && sizes !== null;

    for (let i = 0; i < count; i += 1) {
      const slot = hasSlots ? bindingSlots[i] >>> 0 : 0;
      const handle = hasHandles ? resourceHandles[i] >>> 0 : 0;
      const gen = hasGens ? generations[i] >>> 0 : 0;

      h1 = Math.imul(h1 ^ slot, FNV1A_32_PRIME) >>> 0;
      h1 = Math.imul(h1 ^ handle, FNV1A_32_PRIME) >>> 0;
      h1 = Math.imul(h1 ^ gen, FNV1A_32_PRIME) >>> 0;

      const idxSalt = Math.imul(i + 1, 0x27d4eb2d) >>> 0;
      h2 = Math.imul(h2 ^ splitmix32((slot ^ idxSalt) >>> 0), FNV1A_32_PRIME) >>> 0;
      h2 = Math.imul(h2 ^ splitmix32((handle ^ gen) >>> 0), FNV1A_32_PRIME) >>> 0;
      h2 = ((h2 << 13) | (h2 >>> 19)) >>> 0;
      h2 = Math.imul(h2 ^ gen, FNV1A_32_PRIME) >>> 0;

      if (hasKinds) {
        const k = kinds[i] >>> 0;
        h1 = Math.imul(h1 ^ k, FNV1A_32_PRIME) >>> 0;
        h2 = Math.imul(h2 ^ splitmix32(k), FNV1A_32_PRIME) >>> 0;
      }
      if (hasOffsets) {
        const off = offsets[i];
        if (off !== 0) {
          const offLo = off >>> 0;
          h1 = Math.imul(h1 ^ offLo, FNV1A_32_PRIME) >>> 0;
          h2 = Math.imul(h2 ^ splitmix32(offLo), FNV1A_32_PRIME) >>> 0;
          if (off > 0xffffffff) {
            const offHi = Math.floor(off / TWO_POW_32) >>> 0;
            h1 = Math.imul(h1 ^ offHi, FNV1A_32_PRIME) >>> 0;
            h2 = Math.imul(h2 ^ splitmix32(offHi), FNV1A_32_PRIME) >>> 0;
          }
        }
      }
      if (hasSizes) {
        const sz = sizes[i];
        if (sz !== -1) {
          const szLo = sz >>> 0;
          h1 = Math.imul(h1 ^ szLo, FNV1A_32_PRIME) >>> 0;
          h2 = Math.imul(h2 ^ splitmix32(szLo), FNV1A_32_PRIME) >>> 0;
          if (sz > 0xffffffff) {
            const szHi = Math.floor(sz / TWO_POW_32) >>> 0;
            h1 = Math.imul(h1 ^ szHi, FNV1A_32_PRIME) >>> 0;
            h2 = Math.imul(h2 ^ splitmix32(szHi), FNV1A_32_PRIME) >>> 0;
          }
        }
      }
    }

    const finalLo = splitmix32((h1 ^ (h2 >>> 16)) >>> 0);
    const finalHi = splitmix32((h2 ^ (finalLo >>> 16) ^ 0x517cc1b7) >>> 0);

    if (outPair !== undefined && outPair !== null) {
      outPair[0] = finalLo;
      outPair[1] = finalHi;
    }

    return (finalHi & 0x1fffff) * TWO_POW_32 + finalLo;
  }

  const SLOT_KIND_MAP = Object.freeze({
    buffer: 'buffer',
    gpubuffer: 'buffer',
    uniform: 'buffer',
    storage: 'buffer',
    'read-only-storage': 'buffer',
    texture: 'texture',
    gputexture: 'texture',
    storagetexture: 'texture',
    'storage-texture': 'texture',
    textureview: 'textureView',
    'texture-view': 'textureView',
    texture_view: 'textureView',
    gputextureview: 'textureView',
    sampler: 'sampler',
    gpusampler: 'sampler',
    shadermodule: 'shaderModule',
    'shader-module': 'shaderModule',
    shader_module: 'shaderModule',
    gpushadermodule: 'shaderModule',
    generic: 'generic',
  });

  const KNOWN_SLOT_DESCRIPTOR_KEYS = new Set([
    'kind',
    'handle',
    'id',
    'resourceId',
    'resource',
    'offset',
    'size',
    'format',
    'dimension',
    'aspect',
    'baseMipLevel',
    'mipLevelCount',
    'baseArrayLayer',
    'arrayLayerCount',
    'usage',
    'view',
    'viewParams',
  ]);

  const KNOWN_SLOT_DESCRIPTOR_LOOKUP = Object.create(null);
  for (const _k of KNOWN_SLOT_DESCRIPTOR_KEYS) {
    KNOWN_SLOT_DESCRIPTOR_LOOKUP[_k] = 1;
  }

  const KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP = Object.create(null);
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.format = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.dimension = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.aspect = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.baseMipLevel = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.mipLevelCount = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.baseArrayLayer = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.arrayLayerCount = 1;
  KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP.usage = 1;

  const EMPTY_EXTRA_VIEW_RESULT = Object.freeze({ extraViewParams: null, extraViewKey: '' });
  const EMPTY_VIEW_FORMATS = Object.freeze([]);
  const CANONICAL_SCRATCH_KEYS = [];

  function isValidSlotIndex(slotIndex) {
    return typeof slotIndex === 'number' && Number.isInteger(slotIndex) && slotIndex >= 0;
  }

  function isDirectGpuResource(value) {
    if (!value || typeof value !== 'object') {
      return false;
    }
    if (
      value.kind === 'GPUBuffer' ||
      value.kind === 'GPUTexture' ||
      value.kind === 'GPUTextureView' ||
      value.kind === 'GPUSampler' ||
      value.kind === 'GPUShaderModule'
    ) {
      return true;
    }
    return typeof value.destroy === 'function' || typeof value.createView === 'function';
  }

  function normalizeSlotKind(rawKind, directResource) {
    if (typeof rawKind === 'string' && rawKind.length > 0) {
      const mapped = SLOT_KIND_MAP[rawKind.toLowerCase()];
      if (mapped) {
        return mapped;
      }
      return rawKind.toLowerCase();
    }
    if (directResource && typeof directResource === 'object') {
      if (typeof directResource.kind === 'string') {
        const mapped = SLOT_KIND_MAP[directResource.kind.toLowerCase()];
        if (mapped) {
          return mapped;
        }
      }
      if (typeof directResource.createView === 'function') {
        return 'texture';
      }
      if (typeof directResource.destroy === 'function') {
        return 'buffer';
      }
    }
    return 'generic';
  }

  function sortStringKeysInPlace(keys, count) {
    if (count <= 16) {
      for (let i = 1; i < count; i += 1) {
        const cur = keys[i];
        let j = i - 1;
        while (j >= 0 && keys[j] > cur) {
          keys[j + 1] = keys[j];
          j -= 1;
        }
        keys[j + 1] = cur;
      }
      return;
    }
    keys.length = count;
    keys.sort();
  }

  // Deterministic, collision-free, zero-JSON canonicalizer using type tags and
  // length-prefixed strings/keys so arbitrary characters (delimiters, quotes,
  // colons, control chars) can never collide.
  function canonicalizeExtraValue(value, depth) {
    if (value === null) return 'n';
    if (value === undefined) return 'u';
    const t = typeof value;
    if (t === 'string') {
      return `s${value.length}:${value}`;
    }
    if (t === 'number') {
      return value === 0 && 1 / value < 0 ? 'd-0' : `d${value}`;
    }
    if (t === 'boolean') {
      return value ? 'b1' : 'b0';
    }
    if (t === 'bigint') {
      return `i${value}`;
    }
    if (depth > 4) {
      return '[depth]';
    }
    if (Array.isArray(value)) {
      const len = value.length;
      if (len === 0) return 'a0[]';
      let out = `a${len}[`;
      for (let i = 0; i < len; i += 1) {
        if (i > 0) out += ',';
        out += canonicalizeExtraValue(value[i], depth + 1);
      }
      return `${out}]`;
    }
    if (ArrayBuffer.isView(value)) {
      const ctorName = value.constructor ? value.constructor.name : 'view';
      return `v${ctorName.length}:${ctorName}:${value.byteLength}`;
    }
    if (t === 'object') {
      let count = 0;
      let k0 = '';
      let k1 = '';
      let keys = null;
      for (const k in value) {
        if (!Object.prototype.hasOwnProperty.call(value, k)) continue;
        if (count === 0) {
          k0 = k;
        } else if (count === 1) {
          k1 = k;
        } else {
          if (keys === null) {
            if (depth === 0) {
              keys = CANONICAL_SCRATCH_KEYS;
              keys.length = 0;
            } else {
              keys = [];
            }
            keys[0] = k0;
            keys[1] = k1;
          }
          keys[count] = k;
        }
        count += 1;
      }
      if (count === 0) {
        return 'o0{}';
      }
      if (count === 1) {
        return `o1{k${k0.length}:${k0}=${canonicalizeExtraValue(value[k0], depth + 1)}}`;
      }
      if (count === 2) {
        if (k0 > k1) {
          const tmp = k0;
          k0 = k1;
          k1 = tmp;
        }
        return `o2{k${k0.length}:${k0}=${canonicalizeExtraValue(value[k0], depth + 1)},k${k1.length}:${k1}=${canonicalizeExtraValue(value[k1], depth + 1)}}`;
      }
      sortStringKeysInPlace(keys, count);
      let out = `o${count}{`;
      for (let i = 0; i < count; i += 1) {
        const k = keys[i];
        if (i > 0) out += ',';
        out += `k${k.length}:${k}=${canonicalizeExtraValue(value[k], depth + 1)}`;
      }
      if (keys === CANONICAL_SCRATCH_KEYS) {
        keys.length = 0;
      }
      return `${out}}`;
    }
    const str = String(value);
    return `x${t.length}:${t}:${str.length}:${str}`;
  }

  function extractExtraViewParams(primaryObj, nestedViewObj) {
    let extra = null;
    let extraCount = 0;
    let k0 = '';
    let k1 = '';

    if (primaryObj && typeof primaryObj === 'object') {
      for (const k in primaryObj) {
        if (!Object.prototype.hasOwnProperty.call(primaryObj, k)) continue;
        if (KNOWN_SLOT_DESCRIPTOR_LOOKUP[k] === 1) continue;
        const v = primaryObj[k];
        if (v === undefined) continue;
        if (extra === null) {
          extra = Object.create(null);
        }
        if (extra[k] === undefined) {
          if (extraCount === 0) k0 = k;
          else if (extraCount === 1) k1 = k;
          extraCount += 1;
        }
        extra[k] = v;
      }
    }
    if (nestedViewObj && typeof nestedViewObj === 'object') {
      for (const k in nestedViewObj) {
        if (!Object.prototype.hasOwnProperty.call(nestedViewObj, k)) continue;
        if (KNOWN_SLOT_DESCRIPTOR_LOOKUP[k] === 1) continue;
        const v = nestedViewObj[k];
        if (v === undefined) continue;
        if (extra === null) {
          extra = Object.create(null);
        }
        if (extra[k] === undefined) {
          if (extraCount === 0) k0 = k;
          else if (extraCount === 1) k1 = k;
          extraCount += 1;
        }
        extra[k] = v;
      }
    }
    if (extra === null) {
      return EMPTY_EXTRA_VIEW_RESULT;
    }
    let extraViewKey;
    if (extraCount === 1) {
      extraViewKey = `o1{k${k0.length}:${k0}=${canonicalizeExtraValue(extra[k0], 1)}}`;
    } else if (extraCount === 2) {
      if (k0 > k1) {
        const tmp = k0;
        k0 = k1;
        k1 = tmp;
      }
      extraViewKey = `o2{k${k0.length}:${k0}=${canonicalizeExtraValue(extra[k0], 1)},k${k1.length}:${k1}=${canonicalizeExtraValue(extra[k1], 1)}}`;
    } else {
      extraViewKey = canonicalizeExtraValue(extra, 0);
    }
    return {
      extraViewParams: extra,
      extraViewKey,
    };
  }

  function encodeCanonicalOptionalString(raw, lowercase) {
    if (raw === undefined || raw === null) {
      return 'u';
    }
    if (typeof raw === 'string') {
      if (raw.length === 0) return 'u';
      const s = lowercase ? raw.toLowerCase() : raw;
      return `s${s.length}:${s}`;
    }
    return canonicalizeExtraValue(raw, 0);
  }

  function encodeCanonicalOptionalUint(raw, defaultZero) {
    if (raw === undefined || raw === null) {
      return defaultZero ? 'd0' : 'u';
    }
    const num = Number(raw);
    if (Number.isFinite(num)) {
      return `d${Math.max(0, Math.floor(num))}`;
    }
    return defaultZero ? 'd0' : 'u';
  }

  /**
   * Compute a fast, deterministic, collision-free canonical key for a WebGPU
   * `GPUTextureViewDescriptor` without using `JSON.stringify` or
   * `Object.keys().sort()` on standard descriptors.
   *
   * Known fields (`format`, `dimension`, `aspect`, `baseMipLevel`,
   * `mipLevelCount`, `baseArrayLayer`, `arrayLayerCount`, `usage`) are encoded
   * directly as length-prefixed/tagged scalars; any unknown/extra own keys are
   * deterministically canonicalized only when present.
   *
   * @param {object | null | undefined} viewDesc
   * @returns {string}
   */
  function canonicalizeTextureViewDescriptorKey(viewDesc) {
    if (viewDesc === null || viewDesc === undefined || typeof viewDesc !== 'object') {
      return 'tv:default';
    }
    let hasKnown = false;
    const f = viewDesc.format;
    const d = viewDesc.dimension;
    const a = viewDesc.aspect;
    const bm = viewDesc.baseMipLevel;
    const mc = viewDesc.mipLevelCount;
    const bl = viewDesc.baseArrayLayer;
    const lc = viewDesc.arrayLayerCount;
    const u = viewDesc.usage;
    if (
      (f !== undefined && f !== null && f !== '') ||
      (d !== undefined && d !== null && d !== '') ||
      (a !== undefined && a !== null && a !== '') ||
      (bm !== undefined && bm !== null && Number(bm) > 0) ||
      (mc !== undefined && mc !== null) ||
      (bl !== undefined && bl !== null && Number(bl) > 0) ||
      (lc !== undefined && lc !== null) ||
      (u !== undefined && u !== null)
    ) {
      hasKnown = true;
    }

    // Scan for any unknown own keys without allocating Object.keys().
    let extra = null;
    let extraCount = 0;
    let ek0 = '';
    let ek1 = '';
    for (const k in viewDesc) {
      if (!Object.prototype.hasOwnProperty.call(viewDesc, k)) continue;
      if (KNOWN_TEXTURE_VIEW_DESCRIPTOR_LOOKUP[k] === 1) continue;
      const v = viewDesc[k];
      if (v === undefined) continue;
      if (extra === null) extra = Object.create(null);
      if (extraCount === 0) ek0 = k;
      else if (extraCount === 1) ek1 = k;
      extra[k] = v;
      extraCount += 1;
    }

    if (!hasKnown && extra === null) {
      return 'tv:default';
    }

    const fTok = encodeCanonicalOptionalString(f, true);
    const dTok = encodeCanonicalOptionalString(d, true);
    const aTok = encodeCanonicalOptionalString(a, true);
    const bmTok = encodeCanonicalOptionalUint(bm, true);
    const mcTok = encodeCanonicalOptionalUint(mc, false);
    const blTok = encodeCanonicalOptionalUint(bl, true);
    const lcTok = encodeCanonicalOptionalUint(lc, false);
    const uTok =
      u !== undefined && u !== null && Number.isFinite(Number(u))
        ? `d${Number(u) >>> 0}`
        : 'u';

    const base = `tv|f=${fTok}|d=${dTok}|a=${aTok}|bm=${bmTok}|mc=${mcTok}|bl=${blTok}|lc=${lcTok}|u=${uTok}`;
    if (extra === null) {
      return base;
    }
    let extraKey;
    if (extraCount === 1) {
      extraKey = `o1{k${ek0.length}:${ek0}=${canonicalizeExtraValue(extra[ek0], 1)}}`;
    } else if (extraCount === 2) {
      if (ek0 > ek1) {
        const tmp = ek0;
        ek0 = ek1;
        ek1 = tmp;
      }
      extraKey = `o2{k${ek0.length}:${ek0}=${canonicalizeExtraValue(extra[ek0], 1)},k${ek1.length}:${ek1}=${canonicalizeExtraValue(extra[ek1], 1)}}`;
    } else {
      extraKey = canonicalizeExtraValue(extra, 0);
    }
    return `${base}|x=${extraKey}`;
  }

  /**
   * Compute a fast, deterministic, collision-free canonical key for a
   * `ResourceTable.setSlot` descriptor (supporting flat and nested `view` /
   * `viewParams`, case-insensitive enum strings, and arbitrary property order)
   * without calling `JSON.stringify` or `Object.keys().sort()` on standard
   * descriptors.
   *
   * @param {object | number | null | undefined} descriptorOrHandle
   * @param {object} [viewParamsArg]
   * @returns {string}
   */
  function canonicalizeSlotDescriptorKey(descriptorOrHandle, viewParamsArg) {
    if (descriptorOrHandle === null || descriptorOrHandle === undefined) {
      return 'slot:empty';
    }
    let rawKind;
    let rawHandle = null;
    let directResource = null;
    let srcA = null;
    let srcB = null;

    if (typeof descriptorOrHandle === 'number') {
      rawHandle = descriptorOrHandle;
      if (viewParamsArg && typeof viewParamsArg === 'object') {
        srcA = viewParamsArg;
        rawKind = viewParamsArg.kind;
        srcB = viewParamsArg.view || viewParamsArg.viewParams || null;
      }
    } else if (typeof descriptorOrHandle === 'object') {
      if (isDirectGpuResource(descriptorOrHandle)) {
        directResource = descriptorOrHandle;
        if (viewParamsArg && typeof viewParamsArg === 'object') {
          srcA = viewParamsArg;
          rawKind = viewParamsArg.kind;
          srcB = viewParamsArg.view || viewParamsArg.viewParams || null;
        }
      } else {
        srcA = descriptorOrHandle;
        rawKind = descriptorOrHandle.kind;
        if (descriptorOrHandle.resource && typeof descriptorOrHandle.resource === 'object') {
          directResource = descriptorOrHandle.resource;
        }
        const candidateHandle =
          descriptorOrHandle.handle !== undefined
            ? descriptorOrHandle.handle
            : descriptorOrHandle.id !== undefined
              ? descriptorOrHandle.id
              : descriptorOrHandle.resourceId;
        if (typeof candidateHandle === 'number') {
          rawHandle = candidateHandle;
        } else if (candidateHandle && typeof candidateHandle === 'object') {
          directResource = candidateHandle;
        }
        srcB =
          (viewParamsArg && typeof viewParamsArg === 'object' ? viewParamsArg : null) ||
          descriptorOrHandle.view ||
          descriptorOrHandle.viewParams ||
          null;
      }
    } else {
      return `slot:raw:${canonicalizeExtraValue(descriptorOrHandle, 0)}`;
    }

    const kind = normalizeSlotKind(rawKind, directResource);
    const handle =
      rawHandle !== null && Number.isFinite(rawHandle) ? Math.floor(rawHandle) : null;

    const rawOffset =
      srcB && srcB.offset !== undefined
        ? srcB.offset
        : srcA && srcA.offset !== undefined
          ? srcA.offset
          : undefined;
    const rawSize =
      srcB && srcB.size !== undefined
        ? srcB.size
        : srcA && srcA.size !== undefined
          ? srcA.size
          : undefined;
    const rawFormat =
      srcB && srcB.format !== undefined
        ? srcB.format
        : srcA && srcA.format !== undefined
          ? srcA.format
          : undefined;
    const rawDimension =
      srcB && srcB.dimension !== undefined
        ? srcB.dimension
        : srcA && srcA.dimension !== undefined
          ? srcA.dimension
          : undefined;
    const rawAspect =
      srcB && srcB.aspect !== undefined
        ? srcB.aspect
        : srcA && srcA.aspect !== undefined
          ? srcA.aspect
          : undefined;
    const rawBaseMip =
      srcB && srcB.baseMipLevel !== undefined
        ? srcB.baseMipLevel
        : srcA && srcA.baseMipLevel !== undefined
          ? srcA.baseMipLevel
          : undefined;
    const rawMipCnt =
      srcB && srcB.mipLevelCount !== undefined
        ? srcB.mipLevelCount
        : srcA && srcA.mipLevelCount !== undefined
          ? srcA.mipLevelCount
          : undefined;
    const rawBaseLayer =
      srcB && srcB.baseArrayLayer !== undefined
        ? srcB.baseArrayLayer
        : srcA && srcA.baseArrayLayer !== undefined
          ? srcA.baseArrayLayer
          : undefined;
    const rawLayerCnt =
      srcB && srcB.arrayLayerCount !== undefined
        ? srcB.arrayLayerCount
        : srcA && srcA.arrayLayerCount !== undefined
          ? srcA.arrayLayerCount
          : undefined;
    const rawUsage =
      srcB && srcB.usage !== undefined
        ? srcB.usage
        : srcA && srcA.usage !== undefined
          ? srcA.usage
          : undefined;

    const offTok = encodeCanonicalOptionalUint(rawOffset, true);
    const szTok = encodeCanonicalOptionalUint(rawSize, false);
    const fmtTok = encodeCanonicalOptionalString(rawFormat, true);
    const dimTok = encodeCanonicalOptionalString(rawDimension, true);
    const aspTok = encodeCanonicalOptionalString(rawAspect, true);
    const bmTok = encodeCanonicalOptionalUint(rawBaseMip, true);
    const mcTok = encodeCanonicalOptionalUint(rawMipCnt, false);
    const blTok = encodeCanonicalOptionalUint(rawBaseLayer, true);
    const lcTok = encodeCanonicalOptionalUint(rawLayerCnt, false);
    const usageTok =
      rawUsage !== undefined && rawUsage !== null && Number.isFinite(Number(rawUsage))
        ? `d${Number(rawUsage) >>> 0}`
        : 'u';

    const extraRes = extractExtraViewParams(srcA, srcB);
    const handleTok = handle !== null ? `d${handle}` : directResource !== null ? 'r1' : 'n';
    const base = `slot|k=s${kind.length}:${kind}|h=${handleTok}|o=${offTok}|z=${szTok}|f=${fmtTok}|d=${dimTok}|a=${aspTok}|bm=${bmTok}|mc=${mcTok}|bl=${blTok}|lc=${lcTok}|u=${usageTok}`;
    return extraRes.extraViewKey.length > 0 ? `${base}|x=${extraRes.extraViewKey}` : base;
  }

  function cloneViewDescWithDimension(viewDesc, textureDimension) {
    const safeDesc = {};
    for (const k in viewDesc) {
      if (Object.prototype.hasOwnProperty.call(viewDesc, k)) {
        safeDesc[k] = viewDesc[k];
      }
    }
    safeDesc.dimension = textureDimension;
    return safeDesc;
  }

  // WebGPU-compatible GPUBufferUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_BUFFER_USAGE = Object.freeze({
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

  // WebGPU-compatible GPUTextureUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_TEXTURE_USAGE = Object.freeze({
    COPY_SRC: 0x0001,
    COPY_DST: 0x0002,
    TEXTURE_BINDING: 0x0004,
    STORAGE_BINDING: 0x0008,
    RENDER_ATTACHMENT: 0x0010,
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

  // v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md & texture_packets.js).
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

  // GpuTextureDimension enum names (runtime/abi/playgta6_abi.h).
  const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

  // Valid GPUTextureViewDimension values per GPUTextureDimension per WebGPU spec.
  const VALID_VIEW_DIMENSIONS_BY_TEXTURE_DIMENSION = Object.freeze({
    '1d': new Set(['1d']),
    '2d': new Set(['2d', '2d-array', 'cube', 'cube-array']),
    '3d': new Set(['3d']),
  });

  /**
   * Check whether a requested `GPUTextureViewDimension` is valid for a
   * `GPUTexture` of the given `GPUTextureDimension` per WebGPU validation rules:
   * - `'1d'` texture -> `'1d'` only
   * - `'2d'` texture -> `'2d'`, `'2d-array'`, `'cube'`, `'cube-array'`
   * - `'3d'` texture -> `'3d'` only
   *
   * When `viewDimension` is `undefined` or `null`, returns `true` (WebGPU picks
   * the texture's default compatible view dimension).
   *
   * @param {string | number | null | undefined} textureDimension
   * @param {string | null | undefined} viewDimension
   * @returns {boolean}
   */
  function isViewDimensionCompatible(textureDimension, viewDimension) {
    if (viewDimension === undefined || viewDimension === null) {
      return true;
    }
    if (typeof viewDimension !== 'string') {
      return false;
    }
    let texDim;
    if (typeof textureDimension === 'string') {
      texDim = textureDimension.toLowerCase();
    } else if (typeof textureDimension === 'number' && Number.isInteger(textureDimension)) {
      texDim = TEXTURE_DIMENSION_NAMES[textureDimension];
    } else {
      return false;
    }
    const allowed = VALID_VIEW_DIMENSIONS_BY_TEXTURE_DIMENSION[texDim];
    if (!allowed) {
      return false;
    }
    return allowed.has(viewDimension.toLowerCase());
  }

  // All WebGPU BC1..BC7 4x4 block-compressed texture formats.
  const BC_TEXTURE_FORMATS = Object.freeze([
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc2-rgba-unorm',
    'bc2-rgba-unorm-srgb',
    'bc3-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
    'bc5-rg-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc6h-rgb-float',
    'bc7-rgba-unorm',
    'bc7-rgba-unorm-srgb',
  ]);

  const BC_FORMAT_SET = new Set(BC_TEXTURE_FORMATS);

  function isBcFormat(format) {
    return typeof format === 'string' && BC_FORMAT_SET.has(format.toLowerCase());
  }

  /**
   * Estimate GPU VRAM footprint in bytes for a WebGPU texture across all mip levels,
   * array layers / 3D depth slices, and MSAA samples (#1077 live VRAM accounting).
   */
  function estimateTextureByteSize(
    width,
    height,
    depthOrArrayLayers = 1,
    mipLevelCount = 1,
    format = BC_FALLBACK_FORMAT,
    dimension = '2d',
    sampleCount = 1
  ) {
    const w0 = Math.max(1, Math.floor(Number(width) || 1));
    const h0 = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(height) || 1));
    const d0 = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const mips = Math.max(1, Math.floor(Number(mipLevelCount) || 1));
    const samples = Math.max(1, Math.floor(Number(sampleCount) || 1));
    const fmt = typeof format === 'string' ? format.toLowerCase() : BC_FALLBACK_FORMAT;

    let blockDim = 1;
    let bytesPerBlock = 4;
    if (fmt.startsWith('bc1-') || fmt.startsWith('bc4-')) {
      blockDim = 4;
      bytesPerBlock = 8;
    } else if (
      fmt.startsWith('bc2-') ||
      fmt.startsWith('bc3-') ||
      fmt.startsWith('bc5-') ||
      fmt.startsWith('bc6h-') ||
      fmt.startsWith('bc7-')
    ) {
      blockDim = 4;
      bytesPerBlock = 16;
    } else if (fmt === 'r8unorm' || fmt === 'r8snorm' || fmt === 'r8uint' || fmt === 'r8sint' || fmt === 'stencil8') {
      bytesPerBlock = 1;
    } else if (
      fmt.startsWith('r16') ||
      fmt.startsWith('rg8') ||
      fmt === 'depth16unorm'
    ) {
      bytesPerBlock = 2;
    } else if (
      fmt.startsWith('rg32') ||
      fmt.startsWith('rgba16') ||
      fmt === 'depth32float-stencil8'
    ) {
      bytesPerBlock = 8;
    } else if (fmt.startsWith('rgba32')) {
      bytesPerBlock = 16;
    }

    let total = 0;
    for (let mip = 0; mip < mips; mip += 1) {
      const mw = Math.max(1, Math.floor(w0 / Math.pow(2, mip)));
      const mh = dimension === '1d' ? 1 : Math.max(1, Math.floor(h0 / Math.pow(2, mip)));
      const md = dimension === '3d' ? Math.max(1, Math.floor(d0 / Math.pow(2, mip))) : d0;
      const bw = Math.ceil(mw / blockDim);
      const bh = Math.ceil(mh / blockDim);
      total += bw * bh * md * bytesPerBlock * samples;
    }
    return total;
  }

  /**
   * Check whether a WebGPU device supports `texture-compression-bc`.
   * Returns `false` only when `device.features` exists and does not contain
   * `'texture-compression-bc'`.
   *
   * @param {object | null | undefined} device
   * @returns {boolean}
   */
  function deviceSupportsBc(device) {
    if (!device || !device.features) {
      return true;
    }
    const features = device.features;
    if (typeof features.has === 'function') {
      return Boolean(features.has(TEXTURE_COMPRESSION_BC_FEATURE));
    }
    if (Array.isArray(features)) {
      return features.includes(TEXTURE_COMPRESSION_BC_FEATURE);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === TEXTURE_COMPRESSION_BC_FEATURE) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[TEXTURE_COMPRESSION_BC_FEATURE]);
    }
    return false;
  }

  function resolveTextureFormat(format, useV1Table = false) {
    if (typeof format === 'string' && format.length > 0 && format !== 'undefined') {
      return format.toLowerCase();
    }
    if (typeof format === 'number' && Number.isInteger(format)) {
      if (useV1Table) {
        return V1_TEXTURE_FORMAT_NAMES[format] || BC_FALLBACK_FORMAT;
      }
      const name = TEXTURE_FORMAT_NAMES[format];
      if (format > 0 && name && name !== 'undefined') {
        return name;
      }
    }
    return BC_FALLBACK_FORMAT;
  }

  function resolveTextureDimension(dimension) {
    if (typeof dimension === 'string') {
      const lower = dimension.toLowerCase();
      if (lower === '1d' || lower === '2d' || lower === '3d') {
        return lower;
      }
    }
    if (typeof dimension === 'number' && Number.isInteger(dimension)) {
      const name = TEXTURE_DIMENSION_NAMES[dimension];
      if (name) {
        return name;
      }
    }
    return '2d';
  }

  /**
   * Compute maximum valid WebGPU mip level count for texture dimensions.
   * For 1D/2D/2D-array textures this is `floor(log2(max(width, height))) + 1`;
   * for 3D textures this is `floor(log2(max(width, height, depth))) + 1`.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
    const w = Math.max(1, Math.floor(Number(width) || 1));
    const h = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(height) || 1));
    const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
    return Math.floor(Math.log2(maxDim)) + 1;
  }

  /**
   * Clamp a requested mip level count into `[1, maxMipLevels]`.
   *
   * @param {number | undefined} requestedMipLevels
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {{ mipLevelCount: number, maxMipLevels: number, clamped: boolean }}
   */
  function clampMipLevelCount(
    requestedMipLevels,
    width,
    height,
    depthOrArrayLayers = 1,
    dimension = '2d'
  ) {
    const maxMipLevels = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    if (requestedMipLevels === undefined || requestedMipLevels === null) {
      return { mipLevelCount: 1, maxMipLevels, clamped: false };
    }
    const numeric = Number(requestedMipLevels);
    if (!Number.isFinite(numeric)) {
      return { mipLevelCount: 1, maxMipLevels, clamped: true };
    }
    const truncated = Math.floor(numeric);
    const clampedValue = Math.min(maxMipLevels, Math.max(1, truncated));
    return {
      mipLevelCount: clampedValue,
      maxMipLevels,
      clamped: clampedValue !== numeric,
    };
  }

  /**
   * Clamp a texture `sampleCount` to WebGPU's valid set `{1, 4}`, falling back
   * to `1` whenever `mipLevelCount > 1` or multisampling is unsupported for the
   * texture's format/dimension/layers/usage.
   *
   * @param {number | undefined} requestedSampleCount
   * @param {{
   *   mipLevelCount?: number,
   *   format?: string,
   *   dimension?: string,
   *   depthOrArrayLayers?: number,
   *   usage?: number,
   * }} [context]
   * @returns {{ sampleCount: number, clamped: boolean }}
   */
  function clampSampleCount(requestedSampleCount, context = {}) {
    if (requestedSampleCount === undefined || requestedSampleCount === null) {
      return { sampleCount: 1, clamped: false };
    }
    const numeric = Number(requestedSampleCount);
    if (!Number.isFinite(numeric)) {
      return { sampleCount: 1, clamped: true };
    }

    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const format = context.format || BC_FALLBACK_FORMAT;
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? context.usage >>> 0 : 0;

    const multisampleSupported =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      !isBcFormat(format) &&
      (usage & GPU_TEXTURE_USAGE.STORAGE_BINDING) === 0;

    let resolved;
    if (numeric <= 1) {
      resolved = 1;
    } else if (numeric >= 4 && multisampleSupported) {
      resolved = 4;
    } else {
      // Unsupported MSAA count (e.g. 2x) or texture constraints disallow MSAA.
      resolved = 1;
    }

    return {
      sampleCount: resolved,
      clamped: resolved !== numeric,
    };
  }

  /**
   * Round a single texture dimension up to a multiple of 4 (minimum 4) for BCn blocks.
   *
   * @param {number} dim
   * @returns {number}
   */
  function alignBcDimension(dim) {
    const n = Math.max(1, Math.floor(Number(dim) || 1));
    return Math.ceil(n / 4) * 4;
  }

  /**
   * Align a WebGPU `createBuffer` byte size to 4 bytes, enforcing a minimum of
   * 4 bytes (or 16 bytes when `GPUBufferUsage.UNIFORM` is present).
   * Uses floating-point ceiling rather than 32-bit bitwise ops so sizes > 2 GiB
   * never overflow.
   *
   * @param {number | bigint} requestedSize
   * @param {number} [usage=0]
   * @returns {{ requestedSize: number, alignedSize: number, padded: boolean, isUniform: boolean }}
   */
  function alignBufferSize(requestedSize, usage = 0) {
    const numeric = typeof requestedSize === 'bigint' ? Number(requestedSize) : Number(requestedSize);
    const rawSize = Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 0;
    const usageFlags = Number(usage) >>> 0;
    const isUniform = (usageFlags & GPU_BUFFER_USAGE.UNIFORM) !== 0;
    const minSize = isUniform ? UNIFORM_BUFFER_MIN_BYTES : BUFFER_ALIGNMENT_BYTES;
    const aligned4 = rawSize <= 0 ? BUFFER_ALIGNMENT_BYTES : Math.ceil(rawSize / 4) * 4;
    const alignedSize = Math.max(minSize, aligned4);
    return {
      requestedSize: rawSize,
      alignedSize,
      padded: alignedSize !== rawSize,
      isUniform,
    };
  }

  /**
   * Normalize and 4-byte-align a `queue.writeBuffer` payload so odd-length
   * CBV/VB uploads never fail WebGPU validation.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @param {number} [dataOffset]
   * @param {number} [size]
   * @returns {{ bytes: Uint8Array, originalByteLength: number, alignedByteLength: number, padded: boolean }}
   */
  function padBufferUploadData(data, dataOffset, size) {
    let byteView;
    if (data instanceof Uint8Array) {
      const start = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
      const maxLen = Math.max(0, data.byteLength - start);
      const len = size !== undefined ? Math.min(maxLen, Math.max(0, Math.floor(size))) : maxLen;
      byteView = data.subarray(start, start + len);
    } else if (ArrayBuffer.isView(data)) {
      const elemSize = data.BYTES_PER_ELEMENT || 1;
      const startBytes =
        dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset) * elemSize) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined
          ? Math.min(maxBytes, Math.max(0, Math.floor(size) * elemSize))
          : maxBytes;
      byteView = new Uint8Array(data.buffer, data.byteOffset + startBytes, lenBytes);
    } else if (
      data instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer)
    ) {
      const startBytes = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined ? Math.min(maxBytes, Math.max(0, Math.floor(size))) : maxBytes;
      byteView = new Uint8Array(data, startBytes, lenBytes);
    } else {
      throw new TypeError('writeBuffer data must be an ArrayBufferView or ArrayBuffer');
    }

    const originalByteLength = byteView.byteLength;
    if (originalByteLength === 0) {
      return {
        bytes: new Uint8Array(0),
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
      };
    }

    const remainder = originalByteLength % BUFFER_ALIGNMENT_BYTES;
    if (remainder !== 0) {
      const alignedByteLength = originalByteLength + (BUFFER_ALIGNMENT_BYTES - remainder);
      const paddedCopy = new Uint8Array(alignedByteLength);
      paddedCopy.set(byteView, 0);
      return {
        bytes: paddedCopy,
        originalByteLength,
        alignedByteLength,
        padded: true,
      };
    }

    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && byteView.buffer instanceof SharedArrayBuffer;
    if (isShared) {
      const copy = new Uint8Array(originalByteLength);
      copy.set(byteView, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength: originalByteLength,
        padded: false,
      };
    }

    return {
      bytes: byteView,
      originalByteLength,
      alignedByteLength: originalByteLength,
      padded: false,
    };
  }

  /**
   * Decode a binary ABI v2 `CreateTexture2` payload (opcode 35, 40 bytes,
   * `GpuCreateTexture2Payload` in `runtime/abi/playgta6_abi.h`).
   *
   * Preserves raw `mipLevelCount` and `sampleCount` so `sanitizeTextureDescriptor`
   * can clamp them accurately and update telemetry counters.
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   * @returns {{ id: number, descriptor?: object, error?: string }}
   */
  function decodeCreateTexture2(view, offset = 0, payloadBytes) {
    if (!(view instanceof DataView)) {
      throw new TypeError('decodeCreateTexture2 expects a DataView');
    }
    const available =
      payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (available < CREATE_TEXTURE2_PAYLOAD_BYTES || offset + CREATE_TEXTURE2_PAYLOAD_BYTES > view.byteLength) {
      return {
        id: 0,
        error: `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_PAYLOAD_BYTES} bytes`,
      };
    }

    const id = view.getUint32(offset, true);
    const width = view.getUint32(offset + 4, true);
    const height = view.getUint32(offset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(offset + 12, true));
    const mipLevelCount = view.getUint32(offset + 16, true);
    const sampleCount = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint8(offset + 28);
    const formatCode = view.getUint8(offset + 29);
    const viewFormatCount = view.getUint8(offset + 30);

    const format = TEXTURE_FORMAT_NAMES[formatCode];
    if (formatCode === 0 || format === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}`,
      };
    }

    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
    if (dimension === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}`,
      };
    }

    if (viewFormatCount > MAX_TEXTURE_VIEW_FORMATS) {
      return {
        id,
        error: `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_TEXTURE_VIEW_FORMATS}`,
      };
    }

    const viewFormats = [];
    for (let i = 0; i < viewFormatCount; i += 1) {
      const code = view.getUint8(offset + 32 + i);
      const name = TEXTURE_FORMAT_NAMES[code];
      if (code === 0 || name === undefined) {
        return {
          id,
          error: `CreateTexture2 ${id}: invalid view format code ${code}`,
        };
      }
      viewFormats.push(name);
    }

    const descriptor = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount,
      sampleCount,
      dimension,
      format,
      usage,
    };
    if (viewFormats.length > 0) {
      descriptor.viewFormats = viewFormats;
    }

    return { id, descriptor };
  }

  function extractTextureSize(descriptor, dimension) {
    let rawWidth = 1;
    let rawHeight = 1;
    let rawDepth = 1;

    if (descriptor.size !== undefined && descriptor.size !== null) {
      if (Array.isArray(descriptor.size)) {
        rawWidth = descriptor.size[0] !== undefined ? descriptor.size[0] : 1;
        rawHeight = descriptor.size[1] !== undefined ? descriptor.size[1] : 1;
        rawDepth = descriptor.size[2] !== undefined ? descriptor.size[2] : 1;
      } else if (typeof descriptor.size === 'object') {
        rawWidth = descriptor.size.width !== undefined ? descriptor.size.width : 1;
        rawHeight = descriptor.size.height !== undefined ? descriptor.size.height : 1;
        rawDepth =
          descriptor.size.depthOrArrayLayers !== undefined
            ? descriptor.size.depthOrArrayLayers
            : descriptor.size.depth !== undefined
              ? descriptor.size.depth
              : 1;
      }
    } else {
      rawWidth = descriptor.width !== undefined ? descriptor.width : 1;
      rawHeight = descriptor.height !== undefined ? descriptor.height : 1;
      rawDepth =
        descriptor.depthOrArrayLayers !== undefined
          ? descriptor.depthOrArrayLayers
          : descriptor.depth !== undefined
            ? descriptor.depth
            : descriptor.arrayLayers !== undefined
              ? descriptor.arrayLayers
              : 1;
    }

    const width = Math.max(1, Math.floor(Number(rawWidth) || 1));
    const height = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawHeight) || 1));
    const depthOrArrayLayers =
      dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawDepth) || 1));

    return { width, height, depthOrArrayLayers };
  }

  /**
   * Sanitize a WebGPU texture descriptor (including `CreateTexture2` metadata):
   * - Falls back `BC1..BC7` formats to `'rgba8unorm'` when `device.features`
   *   exists and lacks `'texture-compression-bc'`.
   * - Aligns `BC1..BC7` width/height up to multiples of 4 when BC is supported.
   * - Clamps `mipLevelCount` into `[1, floor(log2(max(width, height))) + 1]`
   *   (or including `depth` for `'3d'`).
   * - Clamps `sampleCount` to `{1, 4}` (falling back to `1` if `mipLevelCount > 1`
   *   or multisampling is unsupported).
   * - Normalizes `viewFormats` (remapping BC view formats on BC fallback).
   *
   * @param {object} rawDescriptor
   * @param {object | null | undefined} [device]
   * @param {{ useV1FormatTable?: boolean, promote1DTo2D?: boolean }} [options]
   * @returns {{
   *   descriptor: object,
   *   originalFormat: string,
   *   effectiveFormat: string,
   *   bcFormatFallback: boolean,
   *   bcDimensionAligned: boolean,
   *   mipClamped: boolean,
   *   sampleClamped: boolean,
   *   promoted1DTo2D: boolean,
   * }}
   */
  function sanitizeTextureDescriptor(rawDescriptor, device, options = {}) {
    if (!rawDescriptor || typeof rawDescriptor !== 'object') {
      throw new TypeError('sanitizeTextureDescriptor expects a descriptor object');
    }

    const rawDimension = resolveTextureDimension(rawDescriptor.dimension);
    const promoted1DTo2D = rawDimension === '1d' && Boolean(options && options.promote1DTo2D);
    const dimension = promoted1DTo2D ? '2d' : rawDimension;
    const originalFormat = resolveTextureFormat(
      rawDescriptor.format,
      Boolean(options && options.useV1FormatTable)
    );
    const usage =
      rawDescriptor.usage !== undefined
        ? Number(rawDescriptor.usage) >>> 0
        : rawDescriptor.usageFlags !== undefined
          ? Number(rawDescriptor.usageFlags) >>> 0
          : GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST;

    let { width, height, depthOrArrayLayers } = extractTextureSize(rawDescriptor, rawDimension);

    let effectiveFormat = originalFormat;
    let bcFormatFallback = false;
    let bcDimensionAligned = false;

    if (isBcFormat(originalFormat)) {
      if (!deviceSupportsBc(device)) {
        effectiveFormat = BC_FALLBACK_FORMAT;
        bcFormatFallback = true;
      } else {
        const alignedWidth = alignBcDimension(width);
        const alignedHeight = dimension === '1d' ? height : alignBcDimension(height);
        if (alignedWidth !== width || alignedHeight !== height) {
          width = alignedWidth;
          height = alignedHeight;
          bcDimensionAligned = true;
        }
      }
    }

    const requestedMipLevels =
      rawDescriptor.mipLevelCount !== undefined
        ? rawDescriptor.mipLevelCount
        : rawDescriptor.mipLevels;
    const mipResult = clampMipLevelCount(
      requestedMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );

    const requestedSampleCount =
      rawDescriptor.sampleCount !== undefined
        ? rawDescriptor.sampleCount
        : rawDescriptor.samples;
    const sampleResult = clampSampleCount(requestedSampleCount, {
      mipLevelCount: mipResult.mipLevelCount,
      format: effectiveFormat,
      dimension,
      depthOrArrayLayers,
      usage,
    });

    const sanitizedDesc = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount: mipResult.mipLevelCount,
      sampleCount: sampleResult.sampleCount,
      dimension,
      format: effectiveFormat,
      usage,
    };

    if (typeof rawDescriptor.label === 'string' && rawDescriptor.label.length > 0) {
      sanitizedDesc.label = rawDescriptor.label;
    }

    if (Array.isArray(rawDescriptor.viewFormats) && rawDescriptor.viewFormats.length > 0) {
      const normalizedViews = [];
      const seen = new Set();
      for (let i = 0; i < rawDescriptor.viewFormats.length; i += 1) {
        const rawVf = rawDescriptor.viewFormats[i];
        let vf =
          typeof rawVf === 'string'
            ? rawVf.toLowerCase()
            : typeof rawVf === 'number' && rawVf > 0
              ? TEXTURE_FORMAT_NAMES[rawVf]
              : undefined;
        if (!vf || vf === 'undefined') {
          continue;
        }
        if (bcFormatFallback && isBcFormat(vf)) {
          vf = vf.endsWith('-srgb') ? BC_FALLBACK_SRGB_FORMAT : BC_FALLBACK_FORMAT;
        }
        if (!seen.has(vf)) {
          seen.add(vf);
          normalizedViews.push(vf);
        }
      }
      if (normalizedViews.length > 0) {
        sanitizedDesc.viewFormats = normalizedViews;
      }
    }

    return {
      descriptor: sanitizedDesc,
      originalFormat,
      effectiveFormat,
      bcFormatFallback,
      bcDimensionAligned,
      mipClamped: mipResult.clamped,
      sampleClamped: sampleResult.clamped,
      promoted1DTo2D,
    };
  }

  /**
   * Hardened WebGPU resource table managing buffers, textures (v1 & ABI v2
   * CreateTexture2), texture views, samplers, and shader modules with
   * automatic BC fallback/alignment, mip/sample clamping, and 4-byte buffer
   * alignment/padding.
   */
  class ResourceTable {
    /**
     * @param {object} device WebGPU GPUDevice
     * @param {{
     *   resourceRegistry?: object | null,
     *   registry?: object | null,
     *   onWarning?: ((event: { kind: string, id: number, detail: string }) => void) | null,
     *   fastSlotCapacity?: number,
     *   maxBindGroups?: number,
     *   deferDestroys?: boolean,
     *   promote1DTo2D?: boolean,
     * }} [options]
     */
    constructor(device, options = {}) {
      if (!device || typeof device !== 'object') {
        throw new TypeError('ResourceTable requires a valid WebGPU device object');
      }
      if (options === null || typeof options !== 'object') {
        throw new TypeError('ResourceTable options must be an object');
      }

      this.device = device;
      this.resourceRegistry = options.resourceRegistry || options.registry || null;
      this._onWarning = typeof options.onWarning === 'function' ? options.onWarning : null;

      // Mid-frame deferred destruction state (W340 / #1459) & W352 destroy/chunk telemetry.
      // Defaults to false so synchronous destroys outside a frame stay 100% backwards compatible.
      this._deferDestroys = Boolean(options.deferDestroys);
      this._inFrame = false;
      this._suppressDeferDestroys = false;
      /** @type {Array<object>} */
      this._deferredDestroys = [];
      /** @type {Array<string>} */
      this._deferredDestroyKinds = [];
      /** @type {Set<object>} */
      this._deferredDestroySet = new Set();
      /** @type {WeakMap<object, number> | null} */
      this._deferredResourceBytes = typeof WeakMap === 'function' ? new WeakMap() : null;
      /** @type {Set<{ items: Array<object>, kinds: Array<string> }>} */
      this._inFlightDeferredBatches = new Set();
      this._deferredDestroyCount = 0;
      this._pendingDeferredDestroyBytes = 0;
      this._destroyedObjects = typeof WeakSet === 'function' ? new WeakSet() : null;
      this._destroyedBuffers = 0;
      this._destroyedTextures = 0;

      // W371 (#1077): Live & peak GPU buffer/texture byte accounting.
      this._liveBufferBytes = 0;
      this._peakBufferBytes = 0;
      this._liveTextureBytes = 0;
      this._peakTextureBytes = 0;

      // W352: Bounded chunk size for queue.writeBuffer to prevent Dawn upload ring spikes.
      const rawMaxWriteChunk =
        options.maxWriteChunkBytes !== undefined
          ? options.maxWriteChunkBytes
          : DEFAULT_MAX_WRITE_CHUNK_BYTES;
      if (!Number.isInteger(rawMaxWriteChunk) || rawMaxWriteChunk < BUFFER_ALIGNMENT_BYTES) {
        throw new RangeError(
          `ResourceTable maxWriteChunkBytes must be an integer >= ${BUFFER_ALIGNMENT_BYTES}, got ${rawMaxWriteChunk}`
        );
      }
      this._maxWriteChunkBytes = Math.ceil(rawMaxWriteChunk / BUFFER_ALIGNMENT_BYTES) * BUFFER_ALIGNMENT_BYTES;
      this._writeScratch = null;
      this._chunkedWrites = 0;

      // Optional 1D -> 2D (width x 1) promotion and view dimension guard telemetry (W340 / #1459).
      this._promote1DTo2D = Boolean(options.promote1DTo2D);
      this._promoted1DTo2DCount = 0;
      this._viewDimensionGuardCount = 0;
      this._scratchSanitizedViewResult = {
        viewDesc: undefined,
        adjusted: false,
        textureDimension: '2d',
      };

      /** @type {Map<number, object>} */
      this.buffers = new Map();
      /** @type {Map<number, object>} */
      this._bufferMeta = new Map();
      /** @type {Map<number, object>} */
      this.textures = new Map();
      /** @type {Map<number, object>} */
      this.textureViews = new Map();
      /** @type {Map<number, object>} */
      this._textureMeta = new Map();
      /** @type {Map<number, object>} */
      this.samplers = new Map();
      /** @type {Map<number, object>} */
      this.shaderModules = new Map();

      // Sanitized hardening counters exposed in getStats().
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;

      // Sanitized lifecycle counters.
      this._buffersCreated = 0;
      this._texturesCreated = 0;
      this._texture2Created = 0;
      this._samplersCreated = 0;
      this._shaderModulesCreated = 0;
      this._bufferWrites = 0;
      this._destroyCount = 0;
      this._errors = 0;
      this._lastError = null;

      // Sanitized occupancy, overwrite & lookup telemetry counters.
      this._peakLiveHandles = 0;
      this._overwrites = 0;
      this._missingLookups = 0;
      this._totalInserts = 0;
      this._totalDeletes = 0;

      // Fast descriptor slot cache & monotonic versioned bind-group invalidation (W207).
      const fastSlotCapacity =
        options.fastSlotCapacity !== undefined
          ? options.fastSlotCapacity
          : DEFAULT_FAST_SLOT_CAPACITY;
      if (!Number.isInteger(fastSlotCapacity) || fastSlotCapacity < 0) {
        throw new RangeError(
          `ResourceTable fastSlotCapacity must be a non-negative integer, got ${fastSlotCapacity}`
        );
      }
      this._fastSlotCapacity = fastSlotCapacity;
      /** @type {Array<object | undefined>} */
      this._fastSlots = [];
      /** @type {Map<number, object>} */
      this._overflowSlots = new Map();
      this._activeSlots = 0;
      this._tableVersion = 0;
      this._slotWrites = 0;
      this._elidedNoOpWrites = 0;
      this._slotLookups = 0;

      // Per-resource monotonic version tracking & reverse slot index for O(1) invalidation.
      this._nextResourceVersion = 1;
      this._resourceVersions = {
        buffer: new Map(),
        texture: new Map(),
        sampler: new Map(),
        shaderModule: new Map(),
      };
      this._resourceSlotRefs = {
        buffer: new Map(),
        texture: new Map(),
        sampler: new Map(),
        shaderModule: new Map(),
        generic: new Map(),
      };

      // Bounded LRU GPUBindGroup cache with generation-stamped stale eviction (W267).
      const maxBindGroups =
        options.maxBindGroups !== undefined
          ? options.maxBindGroups
          : DEFAULT_MAX_BIND_GROUPS;
      if (!Number.isInteger(maxBindGroups) || maxBindGroups < 0) {
        throw new RangeError(
          `ResourceTable maxBindGroups must be a non-negative integer, got ${maxBindGroups}`
        );
      }
      this._maxBindGroups = maxBindGroups;
      /** @type {Map<string, object>} */
      this._bindGroupCache = new Map();
      /** @type {Map<string, Set<string>>} */
      this._bindGroupResourceRefs = new Map();
      this._bindGroupHits = 0;
      this._bindGroupMisses = 0;
      this._staleBindGroupEvictions = 0;
      this._lruBindGroupEvictions = 0;

      // Zero-alloc numeric 64-bit fast-cache & scratch probe arrays for GPUBindGroup lookups (W320).
      const requestedFastBgBuckets =
        options.fastBindGroupBuckets !== undefined
          ? options.fastBindGroupBuckets
          : DEFAULT_FAST_BIND_GROUP_BUCKETS;
      if (!Number.isInteger(requestedFastBgBuckets) || requestedFastBgBuckets < 0) {
        throw new RangeError(
          `ResourceTable fastBindGroupBuckets must be a non-negative integer, got ${requestedFastBgBuckets}`
        );
      }
      this._fastBgBucketCount =
        requestedFastBgBuckets > 0 && maxBindGroups > 0
          ? nextPowerOfTwo(requestedFastBgBuckets)
          : 0;
      this._fastBgMask = this._fastBgBucketCount > 0 ? this._fastBgBucketCount - 1 : 0;
      /** @type {Array<object | null> | null} */
      this._fastBgSlots =
        this._fastBgBucketCount > 0 ? new Array(this._fastBgBucketCount).fill(null) : null;
      this._fastBgReplaceCursor = 0;
      this._bindGroupHashFn =
        typeof options.bindGroupHashFn === 'function' ? options.bindGroupHashFn : null;
      this._probeMaxEntries = MAX_FAST_BIND_GROUP_ENTRIES;
      this._probeBindings = new Uint32Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeKinds = new Uint8Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeHandles = new Uint32Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeGenerations = new Uint32Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeOffsets = new Float64Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeSizes = new Float64Array(MAX_FAST_BIND_GROUP_ENTRIES);
      this._probeHashPair = new Uint32Array(2);
      this._probeEpoch = 0;
      this._zeroAllocKeyProbes = 0;
      this._zeroAllocKeyHits = 0;
      this._hashCollisions = 0;

      // Monotonic per-resource generation stamps across create/overwrite/destroy (W267).
      this._nextResourceGeneration = 1;
      this._resourceGenerations = {
        buffer: new Map(),
        texture: new Map(),
        textureView: new Map(),
        sampler: new Map(),
        shaderModule: new Map(),
      };
      this._liveResourceGenerations = {
        buffer: new Map(),
        texture: new Map(),
        textureView: new Map(),
        sampler: new Map(),
        shaderModule: new Map(),
      };
      /** @type {Map<number, number>} */
      this._textureViewParents = new Map();
      /** @type {Map<number, Set<number>>} */
      this._textureChildViews = new Map();
      this._objectOrigin = typeof WeakMap === 'function' ? new WeakMap() : null;
      this._layoutIds = typeof WeakMap === 'function' ? new WeakMap() : null;
      this._primitiveLayoutIds = new Map();
      this._externalObjectIds = typeof WeakMap === 'function' ? new WeakMap() : null;
      this._nextLayoutId = 1;
      this._nextExternalObjectId = 1;
    }

    _getResourceVersionBucket(kind) {
      if (kind === 'buffer') return 'buffer';
      if (kind === 'texture' || kind === 'textureView') return 'texture';
      if (kind === 'sampler') return 'sampler';
      if (kind === 'shaderModule') return 'shaderModule';
      return null;
    }

    _getResourceVersion(kind, handle) {
      if (handle === null || handle === undefined) {
        return 0;
      }
      const bucket = this._getResourceVersionBucket(kind);
      if (bucket !== null) {
        return this._resourceVersions[bucket].get(handle) || 0;
      }
      return (
        this._resourceVersions.buffer.get(handle) ||
        this._resourceVersions.texture.get(handle) ||
        this._resourceVersions.sampler.get(handle) ||
        this._resourceVersions.shaderModule.get(handle) ||
        0
      );
    }

    _bumpResourceVersion(bucket, id) {
      const ver = this._nextResourceVersion;
      this._nextResourceVersion += 1;
      this._resourceVersions[bucket].set(id, ver);
      return ver;
    }

    _addSlotResourceRef(kind, handle, slotIndex) {
      if (handle === null || handle === undefined) {
        return;
      }
      const bucket = this._getResourceVersionBucket(kind) || 'generic';
      const refMap = this._resourceSlotRefs[bucket];
      let slotSet = refMap.get(handle);
      if (slotSet === undefined) {
        slotSet = new Set();
        refMap.set(handle, slotSet);
      }
      slotSet.add(slotIndex);
    }

    _removeSlotResourceRef(kind, handle, slotIndex) {
      if (handle === null || handle === undefined) {
        return;
      }
      const bucket = this._getResourceVersionBucket(kind) || 'generic';
      const refMap = this._resourceSlotRefs[bucket];
      const slotSet = refMap.get(handle);
      if (slotSet !== undefined) {
        slotSet.delete(slotIndex);
        if (slotSet.size === 0) {
          refMap.delete(handle);
        }
      }
    }

    _invalidateSlotsForResource(bucket, id, newResourceVersion) {
      const typedSet = this._resourceSlotRefs[bucket].get(id);
      const genericSet = this._resourceSlotRefs.generic.get(id);
      if (typedSet === undefined && genericSet === undefined) {
        return 0;
      }
      let invalidated = 0;
      if (typedSet !== undefined) {
        for (const slotIndex of typedSet) {
          const entry = this._getSlotEntry(slotIndex);
          if (entry && entry.bound) {
            entry.resourceVersion = newResourceVersion;
            entry.cachedView = null;
            entry.snapshot = null;
            entry.version += 1;
            invalidated += 1;
          }
        }
      }
      if (genericSet !== undefined) {
        for (const slotIndex of genericSet) {
          const entry = this._getSlotEntry(slotIndex);
          if (entry && entry.bound) {
            entry.resourceVersion = this._getResourceVersion('generic', id);
            entry.cachedView = null;
            entry.snapshot = null;
            entry.version += 1;
            invalidated += 1;
          }
        }
      }
      if (invalidated > 0) {
        this._tableVersion += 1;
      }
      return invalidated;
    }

    _normalizeGenerationBucket(kind) {
      const normalized = normalizeSlotKind(kind, null);
      if (
        normalized === 'buffer' ||
        normalized === 'texture' ||
        normalized === 'textureView' ||
        normalized === 'sampler' ||
        normalized === 'shaderModule'
      ) {
        return normalized;
      }
      return null;
    }

    _resourceRefKey(bucket, handle) {
      return `${bucket}:${handle}`;
    }

    _advanceResourceGeneration(bucket, id, isLive) {
      const gen = this._nextResourceGeneration;
      this._nextResourceGeneration += 1;
      const genMap = this._resourceGenerations[bucket];
      if (genMap.has(id)) {
        genMap.delete(id);
      }
      genMap.set(id, gen);
      const liveMap = this._liveResourceGenerations[bucket];
      if (isLive) {
        liveMap.set(id, gen);
      } else {
        liveMap.delete(id);
        if (genMap.size > liveMap.size + MAX_GENERATION_TOMBSTONES) {
          for (const candidateId of genMap.keys()) {
            if (!liveMap.has(candidateId)) {
              genMap.delete(candidateId);
              break;
            }
          }
        }
      }
      return gen;
    }

    _trackObjectOrigin(obj, kind, handle, generation, parentTextureHandle, parentTextureGeneration) {
      if (!obj || typeof obj !== 'object' || this._objectOrigin === null) {
        return;
      }
      const meta = { kind, handle, generation };
      if (parentTextureHandle !== undefined) {
        meta.parentTextureHandle = parentTextureHandle;
        meta.parentTextureGeneration = parentTextureGeneration;
      }
      this._objectOrigin.set(obj, meta);
    }

    _linkTextureViewParent(viewId, textureId) {
      const prevParent = this._textureViewParents.get(viewId);
      if (prevParent !== undefined && prevParent !== textureId) {
        const prevSet = this._textureChildViews.get(prevParent);
        if (prevSet !== undefined) {
          prevSet.delete(viewId);
          if (prevSet.size === 0) {
            this._textureChildViews.delete(prevParent);
          }
        }
      }
      this._textureViewParents.set(viewId, textureId);
      let childSet = this._textureChildViews.get(textureId);
      if (childSet === undefined) {
        childSet = new Set();
        this._textureChildViews.set(textureId, childSet);
      }
      childSet.add(viewId);
    }

    _unlinkTextureViewParent(viewId) {
      const parentId = this._textureViewParents.get(viewId);
      if (parentId !== undefined) {
        this._textureViewParents.delete(viewId);
        const childSet = this._textureChildViews.get(parentId);
        if (childSet !== undefined) {
          childSet.delete(viewId);
          if (childSet.size === 0) {
            this._textureChildViews.delete(parentId);
          }
        }
      }
      return parentId;
    }

    /**
     * Return the latest monotonic generation stamp for a tracked resource handle
     * (`0` if never created or pruned as an ancient destroyed tombstone).
     * Advances monotonically on every create, overwrite, and destroy.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {number}
     */
    getResourceGeneration(kind, handle) {
      if (handle === null || handle === undefined) {
        return 0;
      }
      const bucket = this._normalizeGenerationBucket(kind);
      if (bucket === null) {
        return 0;
      }
      return this._resourceGenerations[bucket].get(handle) || 0;
    }

    /**
     * Return `true` iff `(kind, handle)` is currently live at `generation`.
     * Returns `false` if the handle was destroyed, overwritten, or never created.
     *
     * @param {string} kind
     * @param {number} handle
     * @param {number} generation
     * @returns {boolean}
     */
    isResourceGenerationCurrent(kind, handle, generation) {
      if (
        handle === null ||
        handle === undefined ||
        typeof generation !== 'number' ||
        !Number.isSafeInteger(generation) ||
        generation < 1
      ) {
        return false;
      }
      const bucket = this._normalizeGenerationBucket(kind);
      if (bucket === null) {
        return false;
      }
      return this._liveResourceGenerations[bucket].get(handle) === generation;
    }

    _getLiveHandleCount() {
      return (
        this.buffers.size +
        this.textures.size +
        this.samplers.size +
        this.shaderModules.size
      );
    }

    _updatePeakLiveHandles() {
      const live = this._getLiveHandleCount();
      if (live > this._peakLiveHandles) {
        this._peakLiveHandles = live;
      }
    }

    _recordInsert(wasOverwrite) {
      this._totalInserts += 1;
      if (wasOverwrite) {
        this._overwrites += 1;
      }
      this._updatePeakLiveHandles();
    }

    _warn(kind, id, detail) {
      if (this._onWarning) {
        this._onWarning({ kind, id, detail });
      }
    }

    _recordError(message) {
      this._errors += 1;
      this._lastError = String(message);
    }

    /**
     * Create and track a WebGPU buffer with 4-byte size alignment (minimum 4
     * bytes, or 16 bytes when `GPUBufferUsage.UNIFORM` is set).
     *
     * Accepts either:
     * - `createBuffer(id, { size, usage, mappedAtCreation, label })`
     * - `createBuffer(id, byteSize, usage)`
     *
     * @param {number} id
     * @param {object | number | bigint} descOrSize
     * @param {number} [usageArg]
     * @returns {object} GPUBuffer
     */
    createBuffer(id, descOrSize, usageArg) {
      let rawSize;
      let usage;
      let mappedAtCreation = false;
      let label = `buffer:${id}`;

      if (descOrSize !== null && typeof descOrSize === 'object') {
        rawSize = descOrSize.size !== undefined ? descOrSize.size : descOrSize.byteSize;
        usage = descOrSize.usage !== undefined ? descOrSize.usage : descOrSize.usageFlags;
        mappedAtCreation = Boolean(descOrSize.mappedAtCreation);
        if (typeof descOrSize.label === 'string' && descOrSize.label.length > 0) {
          label = descOrSize.label;
        }
      } else {
        rawSize = descOrSize;
        usage = usageArg;
      }

      const usageFlags =
        usage !== undefined
          ? Number(usage) >>> 0
          : GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.VERTEX;

      const aligned = alignBufferSize(rawSize, usageFlags);
      if (aligned.padded) {
        this._bufferAlignPadCount += 1;
        this._warn(
          'buffer_size_align',
          id,
          `aligned buffer size ${aligned.requestedSize} -> ${aligned.alignedSize}`
        );
      }

      const existed = this.buffers.has(id);
      if (existed) {
        this._destroyBufferInternal(id, false);
      }

      const createDesc = {
        size: aligned.alignedSize,
        usage: usageFlags,
      };
      if (mappedAtCreation) {
        createDesc.mappedAtCreation = true;
      }
      if (label) {
        createDesc.label = label;
      }

      let buffer;
      try {
        buffer = this.device.createBuffer(createDesc);
      } catch (err) {
        if (existed) {
          this._totalDeletes += 1;
          this._resourceVersions.buffer.delete(id);
          this._invalidateSlotsForResource('buffer', id, 0);
        }
        throw err;
      }
      this.buffers.set(id, buffer);
      this._bufferMeta.set(id, {
        id,
        requestedSize: aligned.requestedSize,
        alignedSize: aligned.alignedSize,
        usage: usageFlags,
        isUniform: aligned.isUniform,
        padded: aligned.padded,
        label,
      });
      this._liveBufferBytes += aligned.alignedSize;
      if (this._liveBufferBytes > this._peakBufferBytes) {
        this._peakBufferBytes = this._liveBufferBytes;
      }
      if (this._deferredResourceBytes !== null && buffer && typeof buffer === 'object') {
        this._deferredResourceBytes.set(buffer, aligned.alignedSize);
      }

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('buffer', id, buffer, {
          byteSize: aligned.alignedSize,
          label,
        });
      }

      const bufGen = this._advanceResourceGeneration('buffer', id, true);
      this._trackObjectOrigin(buffer, 'buffer', id, bufGen);
      this._invalidateBindGroupsForBucketHandle('buffer', id);
      const bufResVer = this._bumpResourceVersion('buffer', id);
      this._invalidateSlotsForResource('buffer', id, bufResVer);

      this._buffersCreated += 1;
      this._recordInsert(existed);
      return buffer;
    }

    _acquireWriteScratch(requiredBytes) {
      if (requiredBytes <= 0) {
        return null;
      }
      if (this._writeScratch && this._writeScratch.byteLength >= requiredBytes) {
        return this._writeScratch;
      }
      if (requiredBytes > this._maxWriteChunkBytes) {
        return null;
      }
      const allocBytes = this._writeScratch
        ? Math.min(this._maxWriteChunkBytes, Math.max(requiredBytes, this._writeScratch.byteLength * 2))
        : requiredBytes;
      this._writeScratch = new Uint8Array(allocBytes);
      return this._writeScratch;
    }

    /**
     * Upload bytes to a tracked buffer (or direct GPUBuffer), automatically
     * padding non-4-byte-aligned payloads into a zero-padded 4-byte-aligned copy
     * and chunking large uploads (> `maxWriteChunkBytes`, default 4 MiB) so Dawn's
     * internal upload ring never spikes to multi-GB.
     *
     * @param {number | object} idOrBuffer
     * @param {number} bufferOffset
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
     * @param {number} [dataOffset]
     * @param {number} [size]
     * @returns {boolean}
     */
    writeBuffer(idOrBuffer, bufferOffset, data, dataOffset, size) {
      let buffer = null;
      let resourceId = -1;
      if (typeof idOrBuffer === 'number') {
        resourceId = idOrBuffer;
        buffer = this.buffers.get(idOrBuffer) || null;
        if (!buffer) {
          this._missingLookups += 1;
        }
      } else if (idOrBuffer && typeof idOrBuffer === 'object') {
        buffer = idOrBuffer;
      }
      if (!buffer) {
        return false;
      }

      const dstOffset = Math.max(0, Math.floor(Number(bufferOffset) || 0));

      let byteView;
      if (data instanceof Uint8Array) {
        const start = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
        const maxLen = Math.max(0, data.byteLength - start);
        const len = size !== undefined ? Math.min(maxLen, Math.max(0, Math.floor(size))) : maxLen;
        byteView = data.subarray(start, start + len);
      } else if (ArrayBuffer.isView(data)) {
        const elemSize = data.BYTES_PER_ELEMENT || 1;
        const startBytes =
          dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset) * elemSize) : 0;
        const maxBytes = Math.max(0, data.byteLength - startBytes);
        const lenBytes =
          size !== undefined
            ? Math.min(maxBytes, Math.max(0, Math.floor(size) * elemSize))
            : maxBytes;
        byteView = new Uint8Array(data.buffer, data.byteOffset + startBytes, lenBytes);
      } else if (
        data instanceof ArrayBuffer ||
        (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer)
      ) {
        const startBytes = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
        const maxBytes = Math.max(0, data.byteLength - startBytes);
        const lenBytes =
          size !== undefined ? Math.min(maxBytes, Math.max(0, Math.floor(size))) : maxBytes;
        byteView = new Uint8Array(data, startBytes, lenBytes);
      } else {
        throw new TypeError('writeBuffer data must be an ArrayBufferView or ArrayBuffer');
      }

      const originalByteLength = byteView.byteLength;
      if (originalByteLength === 0) {
        return true;
      }

      const remainder = originalByteLength % BUFFER_ALIGNMENT_BYTES;
      const padded = remainder !== 0;
      const alignedByteLength = padded
        ? originalByteLength + (BUFFER_ALIGNMENT_BYTES - remainder)
        : originalByteLength;
      const isShared =
        typeof SharedArrayBuffer !== 'undefined' && byteView.buffer instanceof SharedArrayBuffer;

      if (padded) {
        this._bufferAlignPadCount += 1;
        this._warn(
          'buffer_write_pad',
          resourceId,
          `padded writeBuffer ${originalByteLength} -> ${alignedByteLength} bytes`
        );
      }

      const chunkMax = this._maxWriteChunkBytes;
      if (alignedByteLength <= chunkMax) {
        if (!padded && !isShared) {
          this.device.queue.writeBuffer(buffer, dstOffset, byteView);
        } else {
          const scratch = this._acquireWriteScratch(alignedByteLength);
          const dst = scratch
            ? scratch.byteLength === alignedByteLength
              ? scratch
              : scratch.subarray(0, alignedByteLength)
            : new Uint8Array(alignedByteLength);
          dst.set(byteView, 0);
          if (padded) {
            dst.fill(0, originalByteLength, alignedByteLength);
          }
          this.device.queue.writeBuffer(buffer, dstOffset, dst);
        }
      } else {
        for (let off = 0; off < originalByteLength; off += chunkMax) {
          const rawLen = Math.min(chunkMax, originalByteLength - off);
          const chunkRem = rawLen % BUFFER_ALIGNMENT_BYTES;
          const chunkAlignedLen =
            chunkRem === 0 ? rawLen : rawLen + (BUFFER_ALIGNMENT_BYTES - chunkRem);
          const srcSlice = byteView.subarray(off, off + rawLen);
          if (!isShared && chunkRem === 0) {
            this.device.queue.writeBuffer(buffer, dstOffset + off, srcSlice);
          } else {
            const scratch = this._acquireWriteScratch(chunkAlignedLen);
            const dst = scratch
              ? scratch.byteLength === chunkAlignedLen
                ? scratch
                : scratch.subarray(0, chunkAlignedLen)
              : new Uint8Array(chunkAlignedLen);
            dst.set(srcSlice, 0);
            if (chunkRem !== 0) {
              dst.fill(0, rawLen, chunkAlignedLen);
            }
            this.device.queue.writeBuffer(buffer, dstOffset + off, dst);
          }
        }
        this._chunkedWrites += 1;
      }

      this._bufferWrites += 1;
      return true;
    }

    /**
     * Mark the start of a GPU frame. While a frame is active (or whenever the
     * table was constructed with `{ deferDestroys: true }`), buffer/texture
     * destroys and handle overwrites enqueue the old GPU resources into
     * `this._deferredDestroys` instead of calling `.destroy()` synchronously.
     *
     * @returns {boolean}
     */
    beginFrame() {
      this._inFrame = true;
      return true;
    }

    _destroySingleResourceNow(resource, kind) {
      if (!resource || typeof resource.destroy !== 'function') {
        return false;
      }
      if (this._deferredDestroySet.delete(resource)) {
        const queuedBytes =
          this._deferredResourceBytes !== null ? this._deferredResourceBytes.get(resource) || 0 : 0;
        if (queuedBytes > 0) {
          this._pendingDeferredDestroyBytes = Math.max(
            0,
            this._pendingDeferredDestroyBytes - queuedBytes
          );
        }
      }
      if (this._destroyedObjects !== null && this._destroyedObjects.has(resource)) {
        return false;
      }
      if (this._destroyedObjects !== null) {
        this._destroyedObjects.add(resource);
      }
      try {
        resource.destroy();
      } catch (_err) {
        // Ignore errors from already-destroyed or lost WebGPU resources.
      }
      if (kind === 'buffer') {
        this._destroyedBuffers += 1;
      } else if (kind === 'texture') {
        this._destroyedTextures += 1;
      }
      return true;
    }

    _destroyBatchNow(batch) {
      if (!batch || !batch.items || batch.items.length === 0) {
        return 0;
      }
      const items = batch.items.splice(0, batch.items.length);
      const kinds = batch.kinds ? batch.kinds.splice(0, batch.kinds.length) : null;
      let destroyed = 0;
      for (let i = 0; i < items.length; i += 1) {
        const res = items[i];
        const kind = kinds ? kinds[i] : '';
        if (this._destroySingleResourceNow(res, kind)) {
          destroyed += 1;
        }
      }
      return destroyed;
    }

    _destroyOrDefer(resource, kind = '') {
      if (!resource || typeof resource.destroy !== 'function') {
        return false;
      }
      if (this._destroyedObjects !== null && this._destroyedObjects.has(resource)) {
        return false;
      }
      if (!this._suppressDeferDestroys && (this._inFrame || this._deferDestroys)) {
        if (this._deferredDestroySet.has(resource)) {
          return false;
        }
        this._deferredDestroySet.add(resource);
        this._deferredDestroys.push(resource);
        this._deferredDestroyKinds.push(kind);
        this._deferredDestroyCount += 1;
        const queuedBytes =
          this._deferredResourceBytes !== null ? this._deferredResourceBytes.get(resource) || 0 : 0;
        if (queuedBytes > 0) {
          this._pendingDeferredDestroyBytes += queuedBytes;
        }
        return false;
      }
      return this._destroySingleResourceNow(resource, kind);
    }

    _getPendingDeferredDestroyCount() {
      let pending = this._deferredDestroys.length;
      if (this._inFlightDeferredBatches.size > 0) {
        for (const batch of this._inFlightDeferredBatches) {
          pending += batch.items.length;
        }
      }
      return pending;
    }

    /**
     * End the active GPU frame and schedule destruction of resources deferred
     * during the frame.
     *
     * Snapshots and clears the current frame's deferred-destroy list. When
     * `queue && typeof queue.onSubmittedWorkDone === 'function'`, each queued
     * resource is destroyed after `queue.onSubmittedWorkDone()` settles (with
     * `.catch(() => {})` safety). Otherwise the snapshotted batch is destroyed
     * synchronously.
     *
     * @param {object | null | undefined} [queue]
     * @returns {number | Promise<number>}
     */
    endFrame(queue) {
      this._inFrame = false;
      if (this._deferredDestroys.length === 0) {
        return 0;
      }
      const batch = {
        items: this._deferredDestroys,
        kinds: this._deferredDestroyKinds,
      };
      this._deferredDestroys = [];
      this._deferredDestroyKinds = [];

      if (queue && typeof queue.onSubmittedWorkDone === 'function') {
        let workPromise;
        try {
          workPromise = queue.onSubmittedWorkDone();
        } catch (_err) {
          return this._destroyBatchNow(batch);
        }
        if (workPromise && typeof workPromise.then === 'function') {
          this._inFlightDeferredBatches.add(batch);
          const finalize = () => {
            this._inFlightDeferredBatches.delete(batch);
            return this._destroyBatchNow(batch);
          };
          return workPromise.then(finalize, finalize).catch(() => 0);
        }
      }
      return this._destroyBatchNow(batch);
    }

    /**
     * Synchronously drain and destroy every queued and in-flight deferred GPU
     * resource. Safe to call multiple times or while `onSubmittedWorkDone()`
     * promises are still pending (no resource is ever double-destroyed).
     *
     * @returns {number} Number of deferred GPU resources destroyed.
     */
    flushDeferredDestroys() {
      let destroyed = 0;
      if (this._deferredDestroys.length > 0) {
        const batch = {
          items: this._deferredDestroys,
          kinds: this._deferredDestroyKinds,
        };
        this._deferredDestroys = [];
        this._deferredDestroyKinds = [];
        destroyed += this._destroyBatchNow(batch);
      }
      if (this._inFlightDeferredBatches.size > 0) {
        const batches = Array.from(this._inFlightDeferredBatches);
        this._inFlightDeferredBatches.clear();
        for (let i = 0; i < batches.length; i += 1) {
          destroyed += this._destroyBatchNow(batches[i]);
        }
      }
      this._deferredDestroySet.clear();
      this._pendingDeferredDestroyBytes = 0;
      return destroyed;
    }

    /**
     * Release the reusable bounded `writeBuffer` scratch buffer (`_writeScratch`).
     *
     * @returns {number} Bytes released from CPU scratch.
     */
    trimWriteScratch() {
      const released =
        this._writeScratch && typeof this._writeScratch.byteLength === 'number'
          ? this._writeScratch.byteLength
          : 0;
      this._writeScratch = null;
      return released;
    }

    /**
     * Return accurate live/peak GPU buffer, texture, scratch, and deferred-destroy byte accounting (#1077).
     */
    getMemoryStats() {
      const writeScratchBytes =
        this._writeScratch && typeof this._writeScratch.byteLength === 'number'
          ? this._writeScratch.byteLength
          : 0;
      const pendingDeferredDestroys = this._getPendingDeferredDestroyCount();
      return {
        liveBufferBytes: this._liveBufferBytes,
        peakBufferBytes: this._peakBufferBytes,
        liveTextureBytes: this._liveTextureBytes,
        peakTextureBytes: this._peakTextureBytes,
        pendingDeferredDestroys,
        pendingDeferredDestroyBytes: this._pendingDeferredDestroyBytes,
        writeScratchBytes,
        totalLiveBytes: this._liveBufferBytes + this._liveTextureBytes,
      };
    }

    /**
     * Check whether `viewDimension` is valid for a texture of `textureDimension`.
     *
     * @param {string | number | null | undefined} textureDimension
     * @param {string | null | undefined} viewDimension
     * @returns {boolean}
     */
    isViewDimensionCompatible(textureDimension, viewDimension) {
      return isViewDimensionCompatible(textureDimension, viewDimension);
    }

    _resolveActualTextureDimension(texture, textureId) {
      if (texture && typeof texture.dimension === 'string') {
        const d = texture.dimension.toLowerCase();
        if (d === '1d' || d === '2d' || d === '3d') {
          return d;
        }
      }
      if (texture && texture.desc && typeof texture.desc.dimension === 'string') {
        const d = texture.desc.dimension.toLowerCase();
        if (d === '1d' || d === '2d' || d === '3d') {
          return d;
        }
      }
      if (textureId !== undefined && textureId !== null) {
        const meta = this._textureMeta.get(textureId);
        if (meta && typeof meta.dimension === 'string') {
          return meta.dimension;
        }
      }
      return '2d';
    }

    _sanitizeViewDescriptorForTexture(texture, viewDesc, textureId) {
      const textureDimension = this._resolveActualTextureDimension(texture, textureId);
      const out = this._scratchSanitizedViewResult;
      out.textureDimension = textureDimension;
      if (viewDesc === undefined || viewDesc === null || typeof viewDesc !== 'object') {
        out.viewDesc = viewDesc;
        out.adjusted = false;
        return out;
      }
      const requestedDim = viewDesc.dimension;
      if (requestedDim === undefined || requestedDim === null) {
        out.viewDesc = viewDesc;
        out.adjusted = false;
        return out;
      }
      if (isViewDimensionCompatible(textureDimension, requestedDim)) {
        out.viewDesc = viewDesc;
        out.adjusted = false;
        return out;
      }
      const safeDesc = cloneViewDescWithDimension(viewDesc, textureDimension);
      if (textureDimension === '1d') {
        if (safeDesc.baseArrayLayer !== undefined && safeDesc.baseArrayLayer > 0) {
          safeDesc.baseArrayLayer = 0;
        }
        if (safeDesc.arrayLayerCount !== undefined && safeDesc.arrayLayerCount > 1) {
          safeDesc.arrayLayerCount = 1;
        }
      }
      this._viewDimensionGuardCount += 1;
      this._warn(
        'texture_view_dimension_guard',
        typeof textureId === 'number' ? textureId : -1,
        `adjusted incompatible view dimension ${String(requestedDim)} -> ${textureDimension} for ${textureDimension} texture`
      );
      out.viewDesc = safeDesc;
      out.adjusted = true;
      return out;
    }

    /**
     * Create and track a WebGPU texture with full mip/sample clamping and BC
     * feature fallback or 4x4 dimension alignment.
     *
     * @param {number} id
     * @param {object} descriptor
     * @param {{ useV1FormatTable?: boolean, isTexture2?: boolean, promote1DTo2D?: boolean }} [options]
     * @returns {object} GPUTexture
     */
    createTexture(id, descriptor, options = {}) {
      const promote1DTo2D =
        options && options.promote1DTo2D !== undefined
          ? Boolean(options.promote1DTo2D)
          : this._promote1DTo2D;
      const sanitized = sanitizeTextureDescriptor(descriptor, this.device, {
        useV1FormatTable: Boolean(options && options.useV1FormatTable),
        promote1DTo2D,
      });

      if (sanitized.promoted1DTo2D) {
        this._promoted1DTo2DCount += 1;
        this._warn(
          'promote_1d_to_2d',
          id,
          `promoted 1d texture to 2d (${sanitized.descriptor.size.width}x1)`
        );
      }

      if (sanitized.bcFormatFallback) {
        this._bcFormatFallbackCount += 1;
        this._warn(
          'bc_format_fallback',
          id,
          `${sanitized.originalFormat} -> ${sanitized.effectiveFormat}`
        );
      }
      if (sanitized.bcDimensionAligned) {
        this._bcDimensionAlignCount += 1;
        this._warn(
          'bc_dimension_align',
          id,
          `aligned BC dimensions to ${sanitized.descriptor.size.width}x${sanitized.descriptor.size.height}`
        );
      }
      if (sanitized.mipClamped) {
        this._mipClampCount += 1;
        this._warn(
          'mip_clamp',
          id,
          `clamped mipLevelCount to ${sanitized.descriptor.mipLevelCount}`
        );
      }
      if (sanitized.sampleClamped) {
        this._sampleClampCount += 1;
        this._warn(
          'sample_clamp',
          id,
          `clamped sampleCount to ${sanitized.descriptor.sampleCount}`
        );
      }

      const existed = this.textures.has(id);
      this._destroyTextureInternal(id, false);

      let texture;
      try {
        texture = this.device.createTexture(sanitized.descriptor);
      } catch (err) {
        if (existed) {
          this._totalDeletes += 1;
          this._resourceVersions.texture.delete(id);
          this._invalidateSlotsForResource('texture', id, 0);
        }
        throw err;
      }

      let textureView = null;
      if (texture && typeof texture.createView === 'function') {
        try {
          textureView = texture.createView();
        } catch (err) {
          // Ensure newly allocated GPUTexture is not leaked if createView() throws.
          this._destroySingleResourceNow(texture, 'texture');
          if (existed) {
            this._totalDeletes += 1;
            this._resourceVersions.texture.delete(id);
            this._invalidateSlotsForResource('texture', id, 0);
          }
          throw err;
        }
      }

      this.textures.set(id, texture);
      const texGen = this._advanceResourceGeneration('texture', id, true);
      this._trackObjectOrigin(texture, 'texture', id, texGen);

      if (textureView) {
        this.textureViews.set(id, textureView);
        this._linkTextureViewParent(id, id);
        const viewGen = this._advanceResourceGeneration('textureView', id, true);
        this._trackObjectOrigin(textureView, 'textureView', id, viewGen, id, texGen);
      } else {
        this._unlinkTextureViewParent(id);
        this._liveResourceGenerations.textureView.delete(id);
      }

      const estimatedBytes = estimateTextureByteSize(
        sanitized.descriptor.size.width,
        sanitized.descriptor.size.height,
        sanitized.descriptor.size.depthOrArrayLayers,
        sanitized.descriptor.mipLevelCount,
        sanitized.effectiveFormat,
        sanitized.descriptor.dimension,
        sanitized.descriptor.sampleCount
      );
      const meta = {
        id,
        width: sanitized.descriptor.size.width,
        height: sanitized.descriptor.size.height,
        depthOrArrayLayers: sanitized.descriptor.size.depthOrArrayLayers,
        mipLevelCount: sanitized.descriptor.mipLevelCount,
        sampleCount: sanitized.descriptor.sampleCount,
        dimension: sanitized.descriptor.dimension,
        format: sanitized.effectiveFormat,
        originalFormat: sanitized.originalFormat,
        usage: sanitized.descriptor.usage,
        byteSize: estimatedBytes,
        viewFormats:
          sanitized.descriptor.viewFormats && sanitized.descriptor.viewFormats.length > 0
            ? sanitized.descriptor.viewFormats.slice()
            : EMPTY_VIEW_FORMATS,
        bcFormatFallback: sanitized.bcFormatFallback,
        bcDimensionAligned: sanitized.bcDimensionAligned,
        mipClamped: sanitized.mipClamped,
        sampleClamped: sanitized.sampleClamped,
        promoted1DTo2D: sanitized.promoted1DTo2D,
      };
      this._textureMeta.set(id, meta);
      this._liveTextureBytes += estimatedBytes;
      if (this._liveTextureBytes > this._peakTextureBytes) {
        this._peakTextureBytes = this._liveTextureBytes;
      }
      if (this._deferredResourceBytes !== null && texture && typeof texture === 'object') {
        this._deferredResourceBytes.set(texture, estimatedBytes);
      }

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        const label = sanitized.descriptor.label || `texture:${id}`;
        this.resourceRegistry.register('texture', id, texture, {
          byteSize: estimatedBytes,
          label,
        });
        if (textureView) {
          this.resourceRegistry.register('textureView', id, textureView, {
            byteSize: 0,
            label: `textureView:${id}`,
          });
        }
      }

      this._invalidateBindGroupsForBucketHandle('texture', id);
      this._invalidateBindGroupsForBucketHandle('textureView', id);
      const texResVer = this._bumpResourceVersion('texture', id);
      this._invalidateSlotsForResource('texture', id, texResVer);

      this._texturesCreated += 1;
      if (options.isTexture2) {
        this._texture2Created += 1;
      }
      this._recordInsert(existed);
      return texture;
    }

    /**
     * Register or overwrite a tracked handle with an externally created
     * `GPUTexture`. If `id` already holds a different texture, the previous
     * texture is retired/deferred via `_destroyTextureInternal(id, false)`.
     *
     * @param {number} id
     * @param {object} texture
     * @param {{
     *   view?: object | null,
     *   viewDescriptor?: object,
     *   createView?: boolean,
     *   dimension?: string | number,
     *   width?: number,
     *   height?: number,
     *   depthOrArrayLayers?: number,
     *   mipLevelCount?: number,
     *   sampleCount?: number,
     *   format?: string,
     *   usage?: number,
     *   label?: string,
     * }} [options]
     * @returns {object} GPUTexture
     */
    setTexture(id, texture, options = {}) {
      if (!texture || typeof texture !== 'object') {
        throw new TypeError('ResourceTable setTexture requires a texture object');
      }
      const existing = this.textures.get(id);
      if (existing === texture) {
        return texture;
      }

      const opts = options && typeof options === 'object' ? options : {};
      const existed = this.textures.has(id);
      if (existed) {
        this._destroyTextureInternal(id, false);
      }

      this.textures.set(id, texture);
      const texGen = this._advanceResourceGeneration('texture', id, true);
      this._trackObjectOrigin(texture, 'texture', id, texGen);

      const rawDesc = texture.desc && typeof texture.desc === 'object' ? texture.desc : null;
      const dimension = resolveTextureDimension(
        opts.dimension !== undefined
          ? opts.dimension
          : texture.dimension !== undefined
            ? texture.dimension
            : rawDesc && rawDesc.dimension !== undefined
              ? rawDesc.dimension
              : '2d'
      );

      let textureView = null;
      if (opts.view && typeof opts.view === 'object') {
        textureView = opts.view;
      } else if (opts.createView !== false && typeof texture.createView === 'function') {
        if (opts.viewDescriptor !== undefined) {
          const guarded = this._sanitizeViewDescriptorForTexture(texture, opts.viewDescriptor, id);
          textureView =
            guarded.viewDesc !== undefined
              ? texture.createView(guarded.viewDesc)
              : texture.createView();
        } else {
          textureView = texture.createView();
        }
      }

      if (textureView) {
        this.textureViews.set(id, textureView);
        this._linkTextureViewParent(id, id);
        const viewGen = this._advanceResourceGeneration('textureView', id, true);
        this._trackObjectOrigin(textureView, 'textureView', id, viewGen, id, texGen);
      } else {
        this._unlinkTextureViewParent(id);
        this._liveResourceGenerations.textureView.delete(id);
      }

      const rawSize = rawDesc && rawDesc.size && typeof rawDesc.size === 'object' ? rawDesc.size : null;
      const width = Math.max(
        1,
        Math.floor(
          Number(
            opts.width !== undefined
              ? opts.width
              : texture.width !== undefined
                ? texture.width
                : rawSize && rawSize.width !== undefined
                  ? rawSize.width
                  : 1
          ) || 1
        )
      );
      const height =
        dimension === '1d'
          ? 1
          : Math.max(
              1,
              Math.floor(
                Number(
                  opts.height !== undefined
                    ? opts.height
                    : texture.height !== undefined
                      ? texture.height
                      : rawSize && rawSize.height !== undefined
                        ? rawSize.height
                        : 1
                ) || 1
              )
            );
      const depthOrArrayLayers =
        dimension === '1d'
          ? 1
          : Math.max(
              1,
              Math.floor(
                Number(
                  opts.depthOrArrayLayers !== undefined
                    ? opts.depthOrArrayLayers
                    : texture.depthOrArrayLayers !== undefined
                      ? texture.depthOrArrayLayers
                      : rawSize && rawSize.depthOrArrayLayers !== undefined
                        ? rawSize.depthOrArrayLayers
                        : 1
                ) || 1
              )
            );
      const format = resolveTextureFormat(
        opts.format !== undefined
          ? opts.format
          : texture.format !== undefined
            ? texture.format
            : rawDesc && rawDesc.format !== undefined
              ? rawDesc.format
              : BC_FALLBACK_FORMAT
      );
      const usage =
        opts.usage !== undefined
          ? Number(opts.usage) >>> 0
          : texture.usage !== undefined
            ? Number(texture.usage) >>> 0
            : rawDesc && rawDesc.usage !== undefined
              ? Number(rawDesc.usage) >>> 0
              : 0;

      const resolvedMips = Math.max(
        1,
        Math.floor(
          Number(
            opts.mipLevelCount !== undefined
              ? opts.mipLevelCount
              : texture.mipLevelCount !== undefined
                ? texture.mipLevelCount
                : rawDesc && rawDesc.mipLevelCount !== undefined
                  ? rawDesc.mipLevelCount
                  : 1
          ) || 1
        )
      );
      const resolvedSamples = Math.max(
        1,
        Math.floor(
          Number(
            opts.sampleCount !== undefined
              ? opts.sampleCount
              : texture.sampleCount !== undefined
                ? texture.sampleCount
                : rawDesc && rawDesc.sampleCount !== undefined
                  ? rawDesc.sampleCount
                  : 1
          ) || 1
        )
      );
      const estimatedBytes = estimateTextureByteSize(
        width,
        height,
        depthOrArrayLayers,
        resolvedMips,
        format,
        dimension,
        resolvedSamples
      );
      this._textureMeta.set(id, {
        id,
        width,
        height,
        depthOrArrayLayers,
        mipLevelCount: resolvedMips,
        sampleCount: resolvedSamples,
        dimension,
        format,
        originalFormat: format,
        usage,
        byteSize: estimatedBytes,
        viewFormats: EMPTY_VIEW_FORMATS,
        bcFormatFallback: false,
        bcDimensionAligned: false,
        mipClamped: false,
        sampleClamped: false,
        promoted1DTo2D: false,
      });
      this._liveTextureBytes += estimatedBytes;
      if (this._liveTextureBytes > this._peakTextureBytes) {
        this._peakTextureBytes = this._liveTextureBytes;
      }
      if (this._deferredResourceBytes !== null && texture && typeof texture === 'object') {
        this._deferredResourceBytes.set(texture, estimatedBytes);
      }

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        const label = opts.label || texture.label || `texture:${id}`;
        this.resourceRegistry.register('texture', id, texture, {
          byteSize: estimatedBytes,
          label,
        });
        if (textureView) {
          this.resourceRegistry.register('textureView', id, textureView, {
            byteSize: 0,
            label: `textureView:${id}`,
          });
        }
      }

      this._invalidateBindGroupsForBucketHandle('texture', id);
      this._invalidateBindGroupsForBucketHandle('textureView', id);
      const texResVer = this._bumpResourceVersion('texture', id);
      this._invalidateSlotsForResource('texture', id, texResVer);

      this._recordInsert(existed);
      return texture;
    }

    /**
     * Create a texture using ABI v2 `CreateTexture2` metadata (opcode 35).
     * Accepts either:
     * - `createTexture2(id, descriptor, options)` (structured descriptor with v2 format code/name)
     * - `createTexture2(dataView, payloadOffset, payloadBytes, options)` (raw 40-byte payload)
     *
     * @param {number | DataView} idOrView
     * @param {object | number} [descriptorOrOffset]
     * @param {number | object} [payloadBytes]
     * @param {object} [options]
     * @returns {object}
     */
    createTexture2(idOrView, descriptorOrOffset, payloadBytes, options) {
      if (idOrView instanceof DataView) {
        return this.createTexture2FromPacket(
          idOrView,
          typeof descriptorOrOffset === 'number' ? descriptorOrOffset : 0,
          typeof payloadBytes === 'number' ? payloadBytes : undefined,
          payloadBytes && typeof payloadBytes === 'object' ? payloadBytes : options
        );
      }
      const opts =
        payloadBytes && typeof payloadBytes === 'object'
          ? payloadBytes
          : options && typeof options === 'object'
            ? options
            : {};
      return this.createTexture(idOrView, descriptorOrOffset, {
        useV1FormatTable: false,
        isTexture2: true,
        promote1DTo2D: opts.promote1DTo2D,
      });
    }

    /**
     * Decode and create a texture from a binary ABI v2 `CreateTexture2` payload
     * (40 bytes, `GpuCreateTexture2Payload`).
     *
     * @param {DataView} view
     * @param {number} [payloadOffset=0]
     * @param {number} [payloadBytes]
     * @param {{ promote1DTo2D?: boolean }} [options]
     * @returns {{ id: number, texture?: object, view?: object | null, meta?: object, error?: string }}
     */
    createTexture2FromPacket(view, payloadOffset = 0, payloadBytes, options = {}) {
      const decoded = decodeCreateTexture2(view, payloadOffset, payloadBytes);
      if (decoded.error) {
        this._recordError(decoded.error);
        return { id: decoded.id, error: decoded.error };
      }

      const opts = options && typeof options === 'object' ? options : {};
      try {
        const texture = this.createTexture(decoded.id, decoded.descriptor, {
          useV1FormatTable: false,
          isTexture2: true,
          promote1DTo2D: opts.promote1DTo2D,
        });
        return {
          id: decoded.id,
          texture,
          view: this.textureViews.get(decoded.id) || null,
          meta: this._textureMeta.get(decoded.id) || null,
        };
      } catch (err) {
        const msg = `CreateTexture2 ${decoded.id}: ${String(err && err.message ? err.message : err)}`;
        this._recordError(msg);
        return { id: decoded.id, error: msg };
      }
    }

    /**
     * Create or overwrite a tracked `GPUTextureView` handle with monotonic
     * generation stamping and automatic parent-texture lifetime tracking.
     *
     * Accepts either:
     * - `createTextureView(viewId, textureId, viewDescriptor)`
     * - `createTextureView(textureId, viewDescriptor)` (overwrites default view for `textureId`)
     *
     * @param {number} id
     * @param {number | object} [textureIdOrDescriptor]
     * @param {object} [viewDescriptorArg]
     * @returns {object | undefined}
     */
    createTextureView(id, textureIdOrDescriptor, viewDescriptorArg) {
      let textureId = id;
      let viewDesc;
      if (typeof textureIdOrDescriptor === 'number') {
        textureId = textureIdOrDescriptor;
        viewDesc = viewDescriptorArg;
      } else {
        viewDesc = textureIdOrDescriptor;
      }

      const texture = this.textures.get(textureId);
      if (!texture || typeof texture.createView !== 'function') {
        this._missingLookups += 1;
        return undefined;
      }

      const existed = this.textureViews.has(id);
      if (existed) {
        this._advanceResourceGeneration('textureView', id, false);
        this._invalidateBindGroupsForBucketHandle('textureView', id);
      }

      const guarded = this._sanitizeViewDescriptorForTexture(texture, viewDesc, textureId);
      const view =
        guarded.viewDesc !== undefined ? texture.createView(guarded.viewDesc) : texture.createView();
      this.textureViews.set(id, view);
      this._linkTextureViewParent(id, textureId);

      const parentTexGen = this._liveResourceGenerations.texture.get(textureId) || 0;
      const viewGen = this._advanceResourceGeneration('textureView', id, true);
      this._trackObjectOrigin(view, 'textureView', id, viewGen, textureId, parentTexGen);
      this._invalidateBindGroupsForBucketHandle('textureView', id);

      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('textureView', id, view, {
          byteSize: 0,
          label: (viewDesc && viewDesc.label) || `textureView:${id}`,
        });
      }

      const texResVer = this._bumpResourceVersion('texture', textureId);
      this._invalidateSlotsForResource('texture', textureId, texResVer);
      if (id !== textureId) {
        const viewSlotVer = this._bumpResourceVersion('texture', id);
        this._invalidateSlotsForResource('texture', id, viewSlotVer);
      }

      return view;
    }

    createSampler(id, descriptor = {}) {
      const existed = this.samplers.has(id);
      const sampler = this.device.createSampler(descriptor);
      if (existed) {
        this._advanceResourceGeneration('sampler', id, false);
        this._invalidateBindGroupsForBucketHandle('sampler', id);
      }
      this.samplers.set(id, sampler);
      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('sampler', id, sampler, {
          byteSize: 0,
          label: descriptor.label || `sampler:${id}`,
        });
      }
      const sampGen = this._advanceResourceGeneration('sampler', id, true);
      this._trackObjectOrigin(sampler, 'sampler', id, sampGen);
      this._invalidateBindGroupsForBucketHandle('sampler', id);
      const sampResVer = this._bumpResourceVersion('sampler', id);
      this._invalidateSlotsForResource('sampler', id, sampResVer);
      this._samplersCreated += 1;
      this._recordInsert(existed);
      return sampler;
    }

    createShaderModule(id, descriptor) {
      const existed = this.shaderModules.has(id);
      const mod = this.device.createShaderModule(descriptor);
      if (existed) {
        this._advanceResourceGeneration('shaderModule', id, false);
        this._invalidateBindGroupsForBucketHandle('shaderModule', id);
      }
      this.shaderModules.set(id, mod);
      if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
        this.resourceRegistry.register('shaderModule', id, mod, {
          byteSize: 0,
          label: (descriptor && descriptor.label) || `shaderModule:${id}`,
        });
      }
      const modGen = this._advanceResourceGeneration('shaderModule', id, true);
      this._trackObjectOrigin(mod, 'shaderModule', id, modGen);
      this._invalidateBindGroupsForBucketHandle('shaderModule', id);
      const modResVer = this._bumpResourceVersion('shaderModule', id);
      this._invalidateSlotsForResource('shaderModule', id, modResVer);
      this._shaderModulesCreated += 1;
      this._recordInsert(existed);
      return mod;
    }

    getBuffer(id) {
      const value = this.buffers.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getBufferMeta(id) {
      const meta = this._bufferMeta.get(id);
      if (meta === undefined) {
        this._missingLookups += 1;
        return undefined;
      }
      return {
        id: meta.id,
        requestedSize: meta.requestedSize,
        alignedSize: meta.alignedSize,
        usage: meta.usage,
        isUniform: meta.isUniform,
        padded: meta.padded,
        label: meta.label,
      };
    }

    hasBuffer(id) {
      return this.buffers.has(id);
    }

    getTexture(id) {
      const value = this.textures.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getTextureView(id) {
      const value = this.textureViews.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getTextureMeta(id) {
      const meta = this._textureMeta.get(id);
      if (meta === undefined) {
        this._missingLookups += 1;
        return undefined;
      }
      return {
        id: meta.id,
        width: meta.width,
        height: meta.height,
        depthOrArrayLayers: meta.depthOrArrayLayers,
        mipLevelCount: meta.mipLevelCount,
        sampleCount: meta.sampleCount,
        dimension: meta.dimension,
        format: meta.format,
        originalFormat: meta.originalFormat,
        usage: meta.usage,
        byteSize: meta.byteSize,
        viewFormats: meta.viewFormats.length > 0 ? meta.viewFormats.slice() : [],
        bcFormatFallback: meta.bcFormatFallback,
        bcDimensionAligned: meta.bcDimensionAligned,
        mipClamped: meta.mipClamped,
        sampleClamped: meta.sampleClamped,
        promoted1DTo2D: meta.promoted1DTo2D,
      };
    }

    hasTexture(id) {
      return this.textures.has(id);
    }

    getSampler(id) {
      const value = this.samplers.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    getShaderModule(id) {
      const value = this.shaderModules.get(id);
      if (value === undefined) {
        this._missingLookups += 1;
      }
      return value;
    }

    _destroyBufferInternal(id, countDelete) {
      const existing = this.buffers.get(id);
      if (!existing) {
        return false;
      }
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retired = this.resourceRegistry.retire('buffer', id);
        if (!retired && typeof existing.destroy === 'function') {
          this._destroyOrDefer(existing, 'buffer');
        }
      } else if (typeof existing.destroy === 'function') {
        this._destroyOrDefer(existing, 'buffer');
      }
      const prevMeta = this._bufferMeta.get(id);
      if (prevMeta && typeof prevMeta.alignedSize === 'number' && prevMeta.alignedSize > 0) {
        this._liveBufferBytes = Math.max(0, this._liveBufferBytes - prevMeta.alignedSize);
      }
      this.buffers.delete(id);
      this._bufferMeta.delete(id);
      this._destroyCount += 1;
      this._advanceResourceGeneration('buffer', id, false);
      this._invalidateBindGroupsForBucketHandle('buffer', id);
      if (countDelete) {
        this._totalDeletes += 1;
        this._resourceVersions.buffer.delete(id);
        this._invalidateSlotsForResource('buffer', id, 0);
      }
      return true;
    }

    destroyBuffer(id) {
      return this._destroyBufferInternal(id, true);
    }

    _destroyTextureInternal(id, countDelete) {
      const existing = this.textures.get(id);
      if (!existing) {
        if (this.textureViews.delete(id)) {
          this._unlinkTextureViewParent(id);
          this._advanceResourceGeneration('textureView', id, false);
          this._invalidateBindGroupsForBucketHandle('textureView', id);
        }
        this._textureMeta.delete(id);
        return false;
      }
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retiredTex = this.resourceRegistry.retire('texture', id);
        this.resourceRegistry.retire('textureView', id);
        if (!retiredTex && typeof existing.destroy === 'function') {
          this._destroyOrDefer(existing, 'texture');
        }
      } else if (typeof existing.destroy === 'function') {
        this._destroyOrDefer(existing, 'texture');
      }
      const childViews = this._textureChildViews.get(id);
      const childViewIds = childViews ? Array.from(childViews) : [];
      this._textureChildViews.delete(id);

      const prevTexMeta = this._textureMeta.get(id);
      if (prevTexMeta && typeof prevTexMeta.byteSize === 'number' && prevTexMeta.byteSize > 0) {
        this._liveTextureBytes = Math.max(0, this._liveTextureBytes - prevTexMeta.byteSize);
      }
      this.textures.delete(id);
      this.textureViews.delete(id);
      this._textureViewParents.delete(id);
      this._textureMeta.delete(id);
      this._destroyCount += 1;

      this._advanceResourceGeneration('texture', id, false);
      this._advanceResourceGeneration('textureView', id, false);
      this._invalidateBindGroupsForBucketHandle('texture', id);
      this._invalidateBindGroupsForBucketHandle('textureView', id);

      for (let i = 0; i < childViewIds.length; i += 1) {
        const childViewId = childViewIds[i];
        if (childViewId === id) {
          continue;
        }
        this.textureViews.delete(childViewId);
        this._textureViewParents.delete(childViewId);
        if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
          this.resourceRegistry.retire('textureView', childViewId);
        }
        this._advanceResourceGeneration('textureView', childViewId, false);
        this._invalidateBindGroupsForBucketHandle('textureView', childViewId);
      }

      if (countDelete) {
        this._totalDeletes += 1;
        this._resourceVersions.texture.delete(id);
        this._invalidateSlotsForResource('texture', id, 0);
      }
      return true;
    }

    destroyTexture(id) {
      return this._destroyTextureInternal(id, true);
    }

    destroyTextureView(id) {
      if (!this.textureViews.delete(id)) {
        return false;
      }
      this._unlinkTextureViewParent(id);
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        this.resourceRegistry.retire('textureView', id);
      }
      this._advanceResourceGeneration('textureView', id, false);
      this._invalidateBindGroupsForBucketHandle('textureView', id);
      this._invalidateSlotsForResource('texture', id, 0);
      return true;
    }

    destroySampler(id) {
      if (!this.samplers.delete(id)) {
        return false;
      }
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        this.resourceRegistry.retire('sampler', id);
      }
      this._destroyCount += 1;
      this._totalDeletes += 1;
      this._resourceVersions.sampler.delete(id);
      this._advanceResourceGeneration('sampler', id, false);
      this._invalidateBindGroupsForBucketHandle('sampler', id);
      this._invalidateSlotsForResource('sampler', id, 0);
      return true;
    }

    destroy(id) {
      let count = 0;
      if (this.destroyBuffer(id)) count += 1;
      if (this.destroyTexture(id)) count += 1;
      if (this.destroySampler(id)) count += 1;
      if (this.shaderModules.delete(id)) {
        if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
          this.resourceRegistry.retire('shaderModule', id);
        }
        this._destroyCount += 1;
        this._totalDeletes += 1;
        this._resourceVersions.shaderModule.delete(id);
        this._advanceResourceGeneration('shaderModule', id, false);
        this._invalidateBindGroupsForBucketHandle('shaderModule', id);
        this._invalidateSlotsForResource('shaderModule', id, 0);
        count += 1;
      }
      return count;
    }

    // -------------------------------------------------------------------------
    // Fast descriptor slot cache & monotonic versioned bind-group invalidation
    // -------------------------------------------------------------------------

    get tableVersion() {
      return this._tableVersion;
    }

    getTableVersion() {
      return this._tableVersion;
    }

    _getSlotEntry(slotIndex) {
      if (!isValidSlotIndex(slotIndex)) {
        return undefined;
      }
      return slotIndex < this._fastSlotCapacity
        ? this._fastSlots[slotIndex]
        : this._overflowSlots.get(slotIndex);
    }

    /**
     * Fast O(1) monotonic version counter for a single descriptor slot.
     * Returns `0` for slots that have never been written.
     *
     * @param {number} slotIndex
     * @returns {number}
     */
    getSlotVersion(slotIndex) {
      const entry = this._getSlotEntry(slotIndex);
      return entry !== undefined ? entry.version : 0;
    }

    /**
     * Fast O(1) existence check for a bound descriptor slot.
     *
     * @param {number} slotIndex
     * @returns {boolean}
     */
    hasSlot(slotIndex) {
      const entry = this._getSlotEntry(slotIndex);
      return Boolean(entry && entry.bound);
    }

    _buildSlotSnapshot(entry) {
      let frozenExtra = null;
      if (entry.extraViewParams !== null) {
        const copy = {};
        const src = entry.extraViewParams;
        for (const k in src) {
          copy[k] = src[k];
        }
        frozenExtra = Object.freeze(copy);
      }
      return Object.freeze({
        slot: entry.slot,
        version: entry.version,
        kind: entry.kind,
        handle: entry.handle,
        resource: entry.resource,
        resourceVersion: entry.resourceVersion,
        offset: entry.offset,
        size: entry.size,
        format: entry.format,
        dimension: entry.dimension,
        aspect: entry.aspect,
        baseMipLevel: entry.baseMipLevel,
        mipLevelCount: entry.mipLevelCount,
        baseArrayLayer: entry.baseArrayLayer,
        arrayLayerCount: entry.arrayLayerCount,
        usage: entry.usage,
        viewParams: frozenExtra,
      });
    }

    /**
     * Fast O(1) lookup of a bound descriptor slot. Returns a cached frozen
     * snapshot (zero-allocation on repeated reads) or `undefined` if unbound.
     *
     * @param {number} slotIndex
     * @returns {Readonly<object> | undefined}
     */
    getSlot(slotIndex) {
      this._slotLookups += 1;
      const entry = this._getSlotEntry(slotIndex);
      if (!entry || !entry.bound) {
        return undefined;
      }
      if (entry.snapshot === null) {
        entry.snapshot = this._buildSlotSnapshot(entry);
      }
      return entry.snapshot;
    }

    /**
     * Bind or update a descriptor slot with fast no-op write elision.
     *
     * Accepts:
     * - `setSlot(slotIndex, { kind, handle, offset, size, format, dimension, aspect, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount, usage, view, viewParams })`
     * - `setSlot(slotIndex, handleOrGpuResource, viewParams)`
     * - `setSlot(slotIndex, null | undefined)` (unbinds the slot)
     *
     * Increments the slot's monotonic `version` and `tableVersion` ONLY when
     * the normalized bound handle/resource or view parameters actually change.
     * Returns `true` if the slot changed, or `false` if the write was elided.
     *
     * @param {number} slotIndex
     * @param {object | number | null | undefined} descriptorOrHandle
     * @param {object} [viewParamsArg]
     * @returns {boolean}
     */
    setSlot(slotIndex, descriptorOrHandle, viewParamsArg) {
      if (!isValidSlotIndex(slotIndex)) {
        throw new RangeError(
          `ResourceTable slotIndex must be a non-negative integer, got ${slotIndex}`
        );
      }

      if (descriptorOrHandle === null || descriptorOrHandle === undefined) {
        const cleared = this.clearSlot(slotIndex);
        if (cleared) {
          this._slotWrites += 1;
          return true;
        }
        this._elidedNoOpWrites += 1;
        return false;
      }

      let rawKind;
      let rawHandle = null;
      let directResource = null;
      let primaryParams = null;
      let nestedViewParams = null;

      if (typeof descriptorOrHandle === 'number') {
        rawHandle = descriptorOrHandle;
        if (viewParamsArg && typeof viewParamsArg === 'object') {
          primaryParams = viewParamsArg;
          rawKind = viewParamsArg.kind;
          nestedViewParams = viewParamsArg.view || viewParamsArg.viewParams || null;
        }
      } else if (typeof descriptorOrHandle === 'object') {
        if (viewParamsArg !== undefined || isDirectGpuResource(descriptorOrHandle)) {
          if (isDirectGpuResource(descriptorOrHandle)) {
            directResource = descriptorOrHandle;
            if (viewParamsArg && typeof viewParamsArg === 'object') {
              primaryParams = viewParamsArg;
              rawKind = viewParamsArg.kind;
              nestedViewParams = viewParamsArg.view || viewParamsArg.viewParams || null;
            }
          } else {
            primaryParams = descriptorOrHandle;
            rawKind = descriptorOrHandle.kind;
            if (descriptorOrHandle.resource && typeof descriptorOrHandle.resource === 'object') {
              directResource = descriptorOrHandle.resource;
            }
            const candidateHandle =
              descriptorOrHandle.handle !== undefined
                ? descriptorOrHandle.handle
                : descriptorOrHandle.id !== undefined
                  ? descriptorOrHandle.id
                  : descriptorOrHandle.resourceId;
            if (typeof candidateHandle === 'number') {
              rawHandle = candidateHandle;
            } else if (candidateHandle && typeof candidateHandle === 'object') {
              directResource = candidateHandle;
            }
            nestedViewParams =
              (viewParamsArg && typeof viewParamsArg === 'object' ? viewParamsArg : null) ||
              descriptorOrHandle.view ||
              descriptorOrHandle.viewParams ||
              null;
          }
        } else {
          primaryParams = descriptorOrHandle;
          rawKind = descriptorOrHandle.kind;
          if (descriptorOrHandle.resource && typeof descriptorOrHandle.resource === 'object') {
            directResource = descriptorOrHandle.resource;
          }
          const candidateHandle =
            descriptorOrHandle.handle !== undefined
              ? descriptorOrHandle.handle
              : descriptorOrHandle.id !== undefined
                ? descriptorOrHandle.id
                : descriptorOrHandle.resourceId;
          if (typeof candidateHandle === 'number') {
            rawHandle = candidateHandle;
          } else if (candidateHandle && typeof candidateHandle === 'object') {
            directResource = candidateHandle;
          }
          nestedViewParams = descriptorOrHandle.view || descriptorOrHandle.viewParams || null;
        }
      } else {
        throw new TypeError(
          'ResourceTable setSlot expects a descriptor object, numeric handle, GPU resource, or null'
        );
      }

      const kind = normalizeSlotKind(rawKind, directResource);
      const handle =
        rawHandle !== null && Number.isFinite(rawHandle) ? Math.floor(rawHandle) : null;
      const resourceVersion =
        directResource !== null ? 0 : this._getResourceVersion(kind, handle);

      const srcA = primaryParams;
      const srcB = nestedViewParams;

      const rawOffset =
        srcB && srcB.offset !== undefined
          ? srcB.offset
          : srcA && srcA.offset !== undefined
            ? srcA.offset
            : undefined;
      const rawSize =
        srcB && srcB.size !== undefined
          ? srcB.size
          : srcA && srcA.size !== undefined
            ? srcA.size
            : undefined;
      const rawFormat =
        srcB && srcB.format !== undefined
          ? srcB.format
          : srcA && srcA.format !== undefined
            ? srcA.format
            : undefined;
      const rawDimension =
        srcB && srcB.dimension !== undefined
          ? srcB.dimension
          : srcA && srcA.dimension !== undefined
            ? srcA.dimension
            : undefined;
      const rawAspect =
        srcB && srcB.aspect !== undefined
          ? srcB.aspect
          : srcA && srcA.aspect !== undefined
            ? srcA.aspect
            : undefined;
      const rawBaseMip =
        srcB && srcB.baseMipLevel !== undefined
          ? srcB.baseMipLevel
          : srcA && srcA.baseMipLevel !== undefined
            ? srcA.baseMipLevel
            : undefined;
      const rawMipCnt =
        srcB && srcB.mipLevelCount !== undefined
          ? srcB.mipLevelCount
          : srcA && srcA.mipLevelCount !== undefined
            ? srcA.mipLevelCount
            : undefined;
      const rawBaseLayer =
        srcB && srcB.baseArrayLayer !== undefined
          ? srcB.baseArrayLayer
          : srcA && srcA.baseArrayLayer !== undefined
            ? srcA.baseArrayLayer
            : undefined;
      const rawLayerCnt =
        srcB && srcB.arrayLayerCount !== undefined
          ? srcB.arrayLayerCount
          : srcA && srcA.arrayLayerCount !== undefined
            ? srcA.arrayLayerCount
            : undefined;
      const rawUsage =
        srcB && srcB.usage !== undefined
          ? srcB.usage
          : srcA && srcA.usage !== undefined
            ? srcA.usage
            : undefined;

      const offset =
        rawOffset !== undefined && rawOffset !== null && Number.isFinite(Number(rawOffset))
          ? Math.max(0, Math.floor(Number(rawOffset)))
          : 0;
      const size =
        rawSize !== undefined && rawSize !== null && Number.isFinite(Number(rawSize))
          ? Math.max(0, Math.floor(Number(rawSize)))
          : undefined;
      const format =
        typeof rawFormat === 'string' && rawFormat.length > 0
          ? rawFormat.toLowerCase()
          : undefined;
      const dimension =
        typeof rawDimension === 'string' && rawDimension.length > 0
          ? rawDimension.toLowerCase()
          : undefined;
      const aspect =
        typeof rawAspect === 'string' && rawAspect.length > 0
          ? rawAspect.toLowerCase()
          : undefined;
      const baseMipLevel =
        rawBaseMip !== undefined && rawBaseMip !== null && Number.isFinite(Number(rawBaseMip))
          ? Math.max(0, Math.floor(Number(rawBaseMip)))
          : 0;
      const mipLevelCount =
        rawMipCnt !== undefined && rawMipCnt !== null && Number.isFinite(Number(rawMipCnt))
          ? Math.max(0, Math.floor(Number(rawMipCnt)))
          : undefined;
      const baseArrayLayer =
        rawBaseLayer !== undefined &&
        rawBaseLayer !== null &&
        Number.isFinite(Number(rawBaseLayer))
          ? Math.max(0, Math.floor(Number(rawBaseLayer)))
          : 0;
      const arrayLayerCount =
        rawLayerCnt !== undefined &&
        rawLayerCnt !== null &&
        Number.isFinite(Number(rawLayerCnt))
          ? Math.max(0, Math.floor(Number(rawLayerCnt)))
          : undefined;
      const usage =
        rawUsage !== undefined && rawUsage !== null && Number.isFinite(Number(rawUsage))
          ? Number(rawUsage) >>> 0
          : undefined;

      const { extraViewParams, extraViewKey } = extractExtraViewParams(srcA, srcB);

      const useFastArray = slotIndex < this._fastSlotCapacity;
      let entry = useFastArray
        ? this._fastSlots[slotIndex]
        : this._overflowSlots.get(slotIndex);

      if (
        entry !== undefined &&
        entry.bound &&
        entry.kind === kind &&
        entry.handle === handle &&
        entry.resource === directResource &&
        entry.resourceVersion === resourceVersion &&
        entry.offset === offset &&
        entry.size === size &&
        entry.format === format &&
        entry.dimension === dimension &&
        entry.aspect === aspect &&
        entry.baseMipLevel === baseMipLevel &&
        entry.mipLevelCount === mipLevelCount &&
        entry.baseArrayLayer === baseArrayLayer &&
        entry.arrayLayerCount === arrayLayerCount &&
        entry.usage === usage &&
        entry.extraViewKey === extraViewKey
      ) {
        this._elidedNoOpWrites += 1;
        return false;
      }

      const wasBound = Boolean(entry && entry.bound);
      const prevKind = wasBound ? entry.kind : null;
      const prevHandle = wasBound ? entry.handle : null;
      const refChanged = !wasBound || prevKind !== kind || prevHandle !== handle;
      if (wasBound && prevHandle !== null && refChanged) {
        this._removeSlotResourceRef(prevKind, prevHandle, slotIndex);
      }

      if (entry === undefined) {
        entry = {
          slot: slotIndex,
          bound: true,
          version: 1,
          kind,
          handle,
          resource: directResource,
          resourceVersion,
          offset,
          size,
          format,
          dimension,
          aspect,
          baseMipLevel,
          mipLevelCount,
          baseArrayLayer,
          arrayLayerCount,
          usage,
          extraViewParams,
          extraViewKey,
          cachedView: null,
          snapshot: null,
        };
        if (useFastArray) {
          this._fastSlots[slotIndex] = entry;
        } else {
          this._overflowSlots.set(slotIndex, entry);
        }
      } else {
        entry.bound = true;
        entry.version += 1;
        entry.kind = kind;
        entry.handle = handle;
        entry.resource = directResource;
        entry.resourceVersion = resourceVersion;
        entry.offset = offset;
        entry.size = size;
        entry.format = format;
        entry.dimension = dimension;
        entry.aspect = aspect;
        entry.baseMipLevel = baseMipLevel;
        entry.mipLevelCount = mipLevelCount;
        entry.baseArrayLayer = baseArrayLayer;
        entry.arrayLayerCount = arrayLayerCount;
        entry.usage = usage;
        entry.extraViewParams = extraViewParams;
        entry.extraViewKey = extraViewKey;
        entry.cachedView = null;
        entry.snapshot = null;
      }

      if (handle !== null && refChanged) {
        this._addSlotResourceRef(kind, handle, slotIndex);
      }

      if (!wasBound) {
        this._activeSlots += 1;
      }
      this._tableVersion += 1;
      this._slotWrites += 1;
      return true;
    }

    /**
     * Clear a single descriptor slot. Increments the slot's monotonic version
     * and `tableVersion` if the slot was previously bound.
     *
     * @param {number} slotIndex
     * @returns {boolean} `true` if a bound slot was cleared, `false` otherwise.
     */
    clearSlot(slotIndex) {
      const entry = this._getSlotEntry(slotIndex);
      if (!entry || !entry.bound) {
        return false;
      }
      if (entry.handle !== null) {
        this._removeSlotResourceRef(entry.kind, entry.handle, slotIndex);
      }
      entry.bound = false;
      entry.handle = null;
      entry.resource = null;
      entry.resourceVersion = 0;
      entry.offset = 0;
      entry.size = undefined;
      entry.format = undefined;
      entry.dimension = undefined;
      entry.aspect = undefined;
      entry.baseMipLevel = 0;
      entry.mipLevelCount = undefined;
      entry.baseArrayLayer = 0;
      entry.arrayLayerCount = undefined;
      entry.usage = undefined;
      entry.extraViewParams = null;
      entry.extraViewKey = '';
      entry.cachedView = null;
      entry.snapshot = null;
      entry.version += 1;
      this._activeSlots -= 1;
      this._tableVersion += 1;
      return true;
    }

    /**
     * Unbind all active descriptor slots while preserving strictly monotonic
     * per-slot and per-table version counters for downstream bind-group caches.
     *
     * @returns {number} Number of previously active slots cleared.
     */
    clearSlots() {
      if (this._activeSlots === 0) {
        return 0;
      }
      let cleared = 0;
      for (let i = 0; i < this._fastSlots.length; i += 1) {
        const entry = this._fastSlots[i];
        if (entry && entry.bound) {
          entry.bound = false;
          entry.handle = null;
          entry.resource = null;
          entry.resourceVersion = 0;
          entry.offset = 0;
          entry.size = undefined;
          entry.format = undefined;
          entry.dimension = undefined;
          entry.aspect = undefined;
          entry.baseMipLevel = 0;
          entry.mipLevelCount = undefined;
          entry.baseArrayLayer = 0;
          entry.arrayLayerCount = undefined;
          entry.usage = undefined;
          entry.extraViewParams = null;
          entry.extraViewKey = '';
          entry.cachedView = null;
          entry.snapshot = null;
          entry.version += 1;
          cleared += 1;
        }
      }
      if (this._overflowSlots.size > 0) {
        for (const entry of this._overflowSlots.values()) {
          if (entry && entry.bound) {
            entry.bound = false;
            entry.handle = null;
            entry.resource = null;
            entry.resourceVersion = 0;
            entry.offset = 0;
            entry.size = undefined;
            entry.format = undefined;
            entry.dimension = undefined;
            entry.aspect = undefined;
            entry.baseMipLevel = 0;
            entry.mipLevelCount = undefined;
            entry.baseArrayLayer = 0;
            entry.arrayLayerCount = undefined;
            entry.usage = undefined;
            entry.extraViewParams = null;
            entry.extraViewKey = '';
            entry.cachedView = null;
            entry.snapshot = null;
            entry.version += 1;
            cleared += 1;
          }
        }
      }
      this._resourceSlotRefs.buffer.clear();
      this._resourceSlotRefs.texture.clear();
      this._resourceSlotRefs.sampler.clear();
      this._resourceSlotRefs.shaderModule.clear();
      this._resourceSlotRefs.generic.clear();
      this._activeSlots = 0;
      if (cleared > 0) {
        this._tableVersion += 1;
      }
      return cleared;
    }

    /**
     * Resolve the live WebGPU resource bound at `slotIndex`, lazily caching
     * custom `GPUTextureView` objects until the slot or underlying texture changes.
     *
     * @param {number} slotIndex
     * @returns {object | undefined}
     */
    resolveSlot(slotIndex) {
      this._slotLookups += 1;
      const entry = this._getSlotEntry(slotIndex);
      if (!entry || !entry.bound) {
        return undefined;
      }
      if (entry.resource !== null) {
        return entry.resource;
      }
      if (entry.handle === null) {
        return undefined;
      }
      if (entry.kind === 'buffer') {
        return this.buffers.get(entry.handle);
      }
      if (entry.kind === 'texture' || entry.kind === 'textureView') {
        const hasCustomView =
          entry.format !== undefined ||
          entry.dimension !== undefined ||
          entry.aspect !== undefined ||
          entry.baseMipLevel !== 0 ||
          entry.mipLevelCount !== undefined ||
          entry.baseArrayLayer !== 0 ||
          entry.arrayLayerCount !== undefined ||
          entry.usage !== undefined;
        if (hasCustomView) {
          if (entry.cachedView !== null) {
            return entry.cachedView;
          }
          const tex = this.textures.get(entry.handle);
          if (!tex || typeof tex.createView !== 'function') {
            return undefined;
          }
          const viewDesc = {};
          if (entry.format !== undefined) viewDesc.format = entry.format;
          if (entry.dimension !== undefined) viewDesc.dimension = entry.dimension;
          if (entry.aspect !== undefined) viewDesc.aspect = entry.aspect;
          if (entry.baseMipLevel !== 0 || entry.mipLevelCount !== undefined) {
            viewDesc.baseMipLevel = entry.baseMipLevel;
          }
          if (entry.mipLevelCount !== undefined) {
            viewDesc.mipLevelCount = entry.mipLevelCount;
          }
          if (entry.baseArrayLayer !== 0 || entry.arrayLayerCount !== undefined) {
            viewDesc.baseArrayLayer = entry.baseArrayLayer;
          }
          if (entry.arrayLayerCount !== undefined) {
            viewDesc.arrayLayerCount = entry.arrayLayerCount;
          }
          if (entry.usage !== undefined) viewDesc.usage = entry.usage;
          const guarded = this._sanitizeViewDescriptorForTexture(tex, viewDesc, entry.handle);
          entry.cachedView = tex.createView(guarded.viewDesc);
          const texGen = this._liveResourceGenerations.texture.get(entry.handle) || 0;
          this._trackObjectOrigin(entry.cachedView, 'texture', entry.handle, texGen);
          return entry.cachedView;
        }
        return this.textureViews.get(entry.handle);
      }
      if (entry.kind === 'sampler') {
        return this.samplers.get(entry.handle);
      }
      if (entry.kind === 'shaderModule') {
        return this.shaderModules.get(entry.handle);
      }
      return (
        this.buffers.get(entry.handle) ||
        this.textureViews.get(entry.handle) ||
        this.samplers.get(entry.handle) ||
        this.shaderModules.get(entry.handle)
      );
    }

    getSlotResource(slotIndex) {
      return this.resolveSlot(slotIndex);
    }

    /**
     * Build a WebGPU `GPUBindGroupEntry` (`{ binding, resource }`) for `slotIndex`.
     * Returns `null` if the slot is unbound or its underlying resource is missing.
     *
     * @param {number} binding
     * @param {number} slotIndex
     * @returns {{ binding: number, resource: object } | null}
     */
    getBindGroupEntry(binding, slotIndex) {
      const entry = this._getSlotEntry(slotIndex);
      if (!entry || !entry.bound) {
        this._slotLookups += 1;
        return null;
      }
      const res = this.resolveSlot(slotIndex);
      if (!res) {
        return null;
      }
      if (entry.kind === 'buffer') {
        const bufBinding = { buffer: res, offset: entry.offset };
        if (entry.size !== undefined) {
          bufBinding.size = entry.size;
        }
        return { binding, resource: bufBinding };
      }
      return { binding, resource: res };
    }

    /**
     * Capture a compact version stamp for a list of descriptor slots so a
     * downstream bind-group cache can validate in O(1) via `isBindGroupCacheValid`.
     *
     * @param {Array<number> | Uint32Array} slotIndices
     * @returns {{ tableVersion: number, slots: number[], versions: number[] }}
     */
    captureSlotVersions(slotIndices) {
      const len = slotIndices ? slotIndices.length : 0;
      const slots = new Array(len);
      const versions = new Array(len);
      for (let i = 0; i < len; i += 1) {
        const s = slotIndices[i];
        slots[i] = s;
        versions[i] = this.getSlotVersion(s);
      }
      return {
        tableVersion: this._tableVersion,
        slots,
        versions,
      };
    }

    /**
     * O(1) bind-group cache validation against a version stamp returned by
     * `captureSlotVersions()`. Fast-paths on a single `tableVersion` equality
     * check; if an unrelated slot moved `tableVersion`, checks only the stamped
     * slots and fast-forwards `snapshot.tableVersion` on match.
     *
     * @param {{ tableVersion: number, slots: number[], versions: number[] } | null | undefined} snapshot
     * @returns {boolean}
     */
    isSlotSnapshotValid(snapshot) {
      if (!snapshot || typeof snapshot !== 'object') {
        return false;
      }
      if (snapshot.tableVersion === this._tableVersion) {
        return true;
      }
      const slots = snapshot.slots;
      const versions = snapshot.versions;
      if (!Array.isArray(slots) || !Array.isArray(versions) || slots.length !== versions.length) {
        return false;
      }
      for (let i = 0; i < slots.length; i += 1) {
        if (this.getSlotVersion(slots[i]) !== versions[i]) {
          return false;
        }
      }
      snapshot.tableVersion = this._tableVersion;
      return true;
    }

    isBindGroupCacheValid(snapshot) {
      return this.isSlotSnapshotValid(snapshot);
    }

    /**
     * Build a compact deterministic version key string for a list of slots
     * (useful as an LRU bind-group cache key).
     *
     * @param {Array<number> | Uint32Array} slotIndices
     * @returns {string}
     */
    getSlotVersionKey(slotIndices) {
      if (!slotIndices || slotIndices.length === 0) {
        return '';
      }
      let key = '';
      for (let i = 0; i < slotIndices.length; i += 1) {
        const s = slotIndices[i];
        if (i > 0) key += '|';
        key += `${s}:${this.getSlotVersion(s)}`;
      }
      return key;
    }

    // -------------------------------------------------------------------------
    // Generation-stamped bounded LRU GPUBindGroup cache (W267 / #1129 / #1077)
    // -------------------------------------------------------------------------

    get maxBindGroups() {
      return this._maxBindGroups;
    }

    _getNumericLayoutId(layout) {
      if (layout && typeof layout === 'object' && this._layoutIds !== null) {
        let id = this._layoutIds.get(layout);
        if (id === undefined) {
          id = this._nextLayoutId;
          this._nextLayoutId += 1;
          this._layoutIds.set(layout, id);
        }
        return id;
      }
      if (typeof layout === 'string' || typeof layout === 'number') {
        let id = this._primitiveLayoutIds.get(layout);
        if (id === undefined) {
          id = this._nextLayoutId;
          this._nextLayoutId += 1;
          this._primitiveLayoutIds.set(layout, id);
        }
        return id;
      }
      return 0;
    }

    _getLayoutId(layout) {
      if (typeof layout === 'string' || typeof layout === 'number') {
        return `p:${layout}`;
      }
      if (layout && typeof layout === 'object' && this._layoutIds !== null) {
        let id = this._layoutIds.get(layout);
        if (id === undefined) {
          id = this._nextLayoutId;
          this._nextLayoutId += 1;
          this._layoutIds.set(layout, id);
        }
        return `o:${id}`;
      }
      return 'null';
    }

    _normalizeGenerationBucketFast(kind) {
      if (
        kind === 'buffer' ||
        kind === 'texture' ||
        kind === 'textureView' ||
        kind === 'sampler' ||
        kind === 'shaderModule'
      ) {
        return kind;
      }
      return this._normalizeGenerationBucket(kind);
    }

    _detachFastBgSlot(entry) {
      if (!entry) {
        return;
      }
      entry.evicted = true;
      const idx = entry.fastSlotIndex;
      if (idx >= 0 && this._fastBgSlots !== null) {
        const slot = this._fastBgSlots[idx];
        if (slot !== null && slot.cacheEntry === entry) {
          slot.occupied = false;
          slot.cacheEntry = null;
          slot.cacheKey = null;
        }
        entry.fastSlotIndex = -1;
      }
    }

    _getExternalObjectId(obj) {
      if (obj && typeof obj === 'object' && this._externalObjectIds !== null) {
        let id = this._externalObjectIds.get(obj);
        if (id === undefined) {
          id = this._nextExternalObjectId;
          this._nextExternalObjectId += 1;
          this._externalObjectIds.set(obj, id);
        }
        return id;
      }
      return 0;
    }

    _addBindGroupResourceRefs(cacheKey, stamps) {
      for (let i = 0; i < stamps.length; i += 1) {
        const refKey = this._resourceRefKey(stamps[i].kind, stamps[i].handle);
        let keySet = this._bindGroupResourceRefs.get(refKey);
        if (keySet === undefined) {
          keySet = new Set();
          this._bindGroupResourceRefs.set(refKey, keySet);
        }
        keySet.add(cacheKey);
      }
    }

    _removeBindGroupResourceRefs(cacheKey, stamps) {
      for (let i = 0; i < stamps.length; i += 1) {
        const refKey = this._resourceRefKey(stamps[i].kind, stamps[i].handle);
        const keySet = this._bindGroupResourceRefs.get(refKey);
        if (keySet !== undefined) {
          keySet.delete(cacheKey);
          if (keySet.size === 0) {
            this._bindGroupResourceRefs.delete(refKey);
          }
        }
      }
    }

    _evictBindGroupEntry(cacheKey, entry, reason) {
      if (!this._bindGroupCache.has(cacheKey)) {
        return false;
      }
      this._bindGroupCache.delete(cacheKey);
      this._detachFastBgSlot(entry);
      if (entry && Array.isArray(entry.stamps)) {
        this._removeBindGroupResourceRefs(cacheKey, entry.stamps);
      }
      if (reason === 'stale') {
        this._staleBindGroupEvictions += 1;
      } else if (reason === 'lru') {
        this._lruBindGroupEvictions += 1;
      }
      return true;
    }

    _invalidateBindGroupsForBucketHandle(bucket, handle) {
      const refKey = this._resourceRefKey(bucket, handle);
      const keySet = this._bindGroupResourceRefs.get(refKey);
      if (keySet === undefined || keySet.size === 0) {
        return 0;
      }
      const keysToEvict = Array.from(keySet);
      let evicted = 0;
      for (let i = 0; i < keysToEvict.length; i += 1) {
        const cacheKey = keysToEvict[i];
        const entry = this._bindGroupCache.get(cacheKey);
        if (entry !== undefined && this._evictBindGroupEntry(cacheKey, entry, 'stale')) {
          evicted += 1;
        }
      }
      return evicted;
    }

    /**
     * Proactively evict all cached `GPUBindGroup` entries referencing
     * `(kind, handle)` (including child views when `kind` is `'texture'`).
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {number} Number of stale bind groups evicted.
     */
    invalidateBindGroupsForResource(kind, handle) {
      if (handle === null || handle === undefined) {
        return 0;
      }
      const bucket = this._normalizeGenerationBucket(kind);
      if (bucket === null) {
        return 0;
      }
      let evicted = this._invalidateBindGroupsForBucketHandle(bucket, handle);
      if (bucket === 'texture') {
        evicted += this._invalidateBindGroupsForBucketHandle('textureView', handle);
        const childViews = this._textureChildViews.get(handle);
        if (childViews !== undefined) {
          for (const childViewId of Array.from(childViews)) {
            if (childViewId !== handle) {
              evicted += this._invalidateBindGroupsForBucketHandle('textureView', childViewId);
            }
          }
        }
      }
      return evicted;
    }

    _isBindGroupEntryValid(entry) {
      if (!entry) {
        return false;
      }
      const stamps = entry.stamps;
      for (let i = 0; i < stamps.length; i += 1) {
        const stamp = stamps[i];
        const liveMap = this._liveResourceGenerations[stamp.kind];
        if (!liveMap || liveMap.get(stamp.handle) !== stamp.generation) {
          return false;
        }
      }
      if (entry.directResources && entry.directResources.length > 0) {
        for (let i = 0; i < entry.directResources.length; i += 1) {
          const res = entry.directResources[i];
          if (!res || res.destroyed === true) {
            return false;
          }
          if (res.buffer && res.buffer.destroyed === true) {
            return false;
          }
          if (this._objectOrigin !== null) {
            const origin = this._objectOrigin.get(res);
            if (origin !== undefined) {
              const liveMap = this._liveResourceGenerations[origin.kind];
              if (!liveMap || liveMap.get(origin.handle) !== origin.generation) {
                return false;
              }
              if (origin.parentTextureHandle !== undefined) {
                if (
                  this._liveResourceGenerations.texture.get(origin.parentTextureHandle) !==
                  origin.parentTextureGeneration
                ) {
                  return false;
                }
              }
            }
          }
        }
      }
      if (entry.slotSnapshot && !this.isSlotSnapshotValid(entry.slotSnapshot)) {
        return false;
      }
      return true;
    }

    _appendStamp(stamps, seenKeys, kind, handle, generation, directResources) {
      const bucket = this._normalizeGenerationBucket(kind);
      if (bucket === null || handle === null || handle === undefined) {
        return;
      }
      const refKey = this._resourceRefKey(bucket, handle);
      if (!seenKeys.has(refKey)) {
        seenKeys.add(refKey);
        stamps.push({ kind: bucket, handle, generation });
      }
      if (directResources) {
        const liveGen = this._liveResourceGenerations[bucket].get(handle);
        if (liveGen !== undefined && liveGen === generation) {
          if (bucket === 'buffer') {
            const b = this.buffers.get(handle);
            if (b) directResources.push(b);
          } else if (bucket === 'texture') {
            const t = this.textures.get(handle);
            if (t) directResources.push(t);
          } else if (bucket === 'textureView') {
            const v = this.textureViews.get(handle);
            if (v) directResources.push(v);
          } else if (bucket === 'sampler') {
            const s = this.samplers.get(handle);
            if (s) directResources.push(s);
          }
        }
      }
      if (bucket === 'textureView') {
        const parentTexId = this._textureViewParents.get(handle);
        if (parentTexId !== undefined) {
          const parentKey = this._resourceRefKey('texture', parentTexId);
          if (!seenKeys.has(parentKey)) {
            const parentGen = this._liveResourceGenerations.texture.get(parentTexId) || 0;
            seenKeys.add(parentKey);
            stamps.push({ kind: 'texture', handle: parentTexId, generation: parentGen });
            if (directResources) {
              const parentTex = this.textures.get(parentTexId);
              if (parentTex) directResources.push(parentTex);
            }
          }
        }
      }
    }

    _normalizeBindGroupStamps(resourceRefs, requireLive) {
      const stamps = [];
      const seenKeys = new Set();
      const directResources = [];
      const slotIndices = [];

      if (!Array.isArray(resourceRefs)) {
        return { ok: true, stamps, directResources, slotSnapshot: null };
      }

      for (let i = 0; i < resourceRefs.length; i += 1) {
        const ref = resourceRefs[i];
        if (ref === null || ref === undefined) {
          continue;
        }
        if (Array.isArray(ref)) {
          const bucket = this._normalizeGenerationBucket(ref[0]);
          const handle = ref[1];
          if (bucket === null || typeof handle !== 'number') {
            continue;
          }
          const explicitGen = ref.length >= 3 && typeof ref[2] === 'number' ? ref[2] : undefined;
          const liveGen = this._liveResourceGenerations[bucket].get(handle);
          if (requireLive && (liveGen === undefined || (explicitGen !== undefined && explicitGen !== liveGen))) {
            this._missingLookups += 1;
            return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
          }
          const gen = explicitGen !== undefined ? explicitGen : liveGen !== undefined ? liveGen : 0;
          this._appendStamp(stamps, seenKeys, bucket, handle, gen, directResources);
          continue;
        }
        if (typeof ref === 'object') {
          if (ref.slot !== undefined && isValidSlotIndex(ref.slot)) {
            const slotEntry = this._getSlotEntry(ref.slot);
            if (!slotEntry || !slotEntry.bound) {
              if (requireLive) {
                this._missingLookups += 1;
                return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
              }
              continue;
            }
            slotIndices.push(ref.slot);
            if (slotEntry.resource !== null) {
              if (requireLive && slotEntry.resource.destroyed === true) {
                return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
              }
              directResources.push(slotEntry.resource);
              if (this._objectOrigin !== null) {
                const origin = this._objectOrigin.get(slotEntry.resource);
                if (origin !== undefined) {
                  if (requireLive && !this.isResourceGenerationCurrent(origin.kind, origin.handle, origin.generation)) {
                    return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
                  }
                  this._appendStamp(stamps, seenKeys, origin.kind, origin.handle, origin.generation, null);
                }
              }
            } else if (slotEntry.handle !== null) {
              const bucket = this._normalizeGenerationBucket(slotEntry.kind);
              if (bucket !== null) {
                const liveGen = this._liveResourceGenerations[bucket].get(slotEntry.handle);
                if (requireLive && liveGen === undefined) {
                  this._missingLookups += 1;
                  return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
                }
                this._appendStamp(
                  stamps,
                  seenKeys,
                  bucket,
                  slotEntry.handle,
                  liveGen !== undefined ? liveGen : 0,
                  directResources
                );
                if (bucket === 'texture') {
                  const viewGen = this._liveResourceGenerations.textureView.get(slotEntry.handle);
                  if (viewGen !== undefined) {
                    this._appendStamp(stamps, seenKeys, 'textureView', slotEntry.handle, viewGen, directResources);
                  }
                }
              }
            }
            continue;
          }

          if (ref.handle !== undefined || ref.id !== undefined || ref.resourceId !== undefined) {
            const handle =
              ref.handle !== undefined
                ? ref.handle
                : ref.id !== undefined
                  ? ref.id
                  : ref.resourceId;
            const bucket = this._normalizeGenerationBucket(ref.kind);
            if (bucket !== null && typeof handle === 'number') {
              const explicitGen = typeof ref.generation === 'number' ? ref.generation : undefined;
              const liveGen = this._liveResourceGenerations[bucket].get(handle);
              if (requireLive && (liveGen === undefined || (explicitGen !== undefined && explicitGen !== liveGen))) {
                this._missingLookups += 1;
                return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
              }
              const gen = explicitGen !== undefined ? explicitGen : liveGen !== undefined ? liveGen : 0;
              this._appendStamp(stamps, seenKeys, bucket, handle, gen, directResources);
              if (bucket === 'texture') {
                const viewGen = this._liveResourceGenerations.textureView.get(handle);
                if (viewGen !== undefined) {
                  this._appendStamp(stamps, seenKeys, 'textureView', handle, viewGen, directResources);
                }
              }
              continue;
            }
          }

          // Direct GPU resource object (or { buffer } binding wrapper).
          const targetObj = ref.buffer && typeof ref.buffer === 'object' ? ref.buffer : ref;
          if (requireLive && targetObj.destroyed === true) {
            return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
          }
          directResources.push(targetObj);
          if (this._objectOrigin !== null) {
            const origin = this._objectOrigin.get(targetObj);
            if (origin !== undefined) {
              if (requireLive && !this.isResourceGenerationCurrent(origin.kind, origin.handle, origin.generation)) {
                return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
              }
              if (
                requireLive &&
                origin.parentTextureHandle !== undefined &&
                !this.isResourceGenerationCurrent(
                  'texture',
                  origin.parentTextureHandle,
                  origin.parentTextureGeneration
                )
              ) {
                return { ok: false, stamps: [], directResources: [], slotSnapshot: null };
              }
              this._appendStamp(stamps, seenKeys, origin.kind, origin.handle, origin.generation, null);
              if (origin.parentTextureHandle !== undefined) {
                this._appendStamp(
                  stamps,
                  seenKeys,
                  'texture',
                  origin.parentTextureHandle,
                  origin.parentTextureGeneration,
                  null
                );
              }
            }
          }
        }
      }

      const slotSnapshot = slotIndices.length > 0 ? this.captureSlotVersions(slotIndices) : null;
      return { ok: true, stamps, directResources, slotSnapshot };
    }

    _storeBindGroupEntryRecord(cacheKey, bindGroup, stamps, directResources, slotSnapshot) {
      if (this._maxBindGroups === 0) {
        return null;
      }

      const existing = this._bindGroupCache.get(cacheKey);
      if (existing !== undefined) {
        this._detachFastBgSlot(existing);
        this._removeBindGroupResourceRefs(cacheKey, existing.stamps);
        this._bindGroupCache.delete(cacheKey);
      } else {
        while (this._bindGroupCache.size >= this._maxBindGroups) {
          const oldestKey = this._bindGroupCache.keys().next().value;
          const oldestEntry = this._bindGroupCache.get(oldestKey);
          this._evictBindGroupEntry(oldestKey, oldestEntry, 'lru');
        }
      }

      const entry = {
        key: cacheKey,
        bindGroup,
        stamps,
        directResources,
        slotSnapshot,
        fastSlotIndex: -1,
        evicted: false,
      };
      this._bindGroupCache.set(cacheKey, entry);
      this._addBindGroupResourceRefs(cacheKey, stamps);
      return entry;
    }

    _storeBindGroupEntry(cacheKey, bindGroup, stamps, directResources, slotSnapshot) {
      return (
        this._storeBindGroupEntryRecord(cacheKey, bindGroup, stamps, directResources, slotSnapshot) !==
        null
      );
    }

    /**
     * Store a `GPUBindGroup` in the bounded LRU cache stamped with the exact
     * `(handle, generation)` tuples of its referenced resources.
     *
     * @param {string | number} key
     * @param {object} bindGroup
     * @param {Array<object | Array>} [resourceRefs]
     * @returns {boolean}
     */
    setBindGroup(key, bindGroup, resourceRefs = []) {
      if (key === null || key === undefined) {
        throw new TypeError('ResourceTable setBindGroup requires a non-null cache key');
      }
      if (!bindGroup || typeof bindGroup !== 'object') {
        throw new TypeError('ResourceTable setBindGroup requires a bindGroup object');
      }
      const cacheKey = String(key);
      const normalized = this._normalizeBindGroupStamps(resourceRefs, false);
      return this._storeBindGroupEntry(
        cacheKey,
        bindGroup,
        normalized.stamps,
        normalized.directResources,
        normalized.slotSnapshot
      );
    }

    /**
     * Look up a cached `GPUBindGroup` by key. Automatically validates every
     * stamped `(handle, generation)` tuple and direct resource state; stale
     * entries are evicted immediately (`staleBindGroupEvictions += 1`).
     * Valid hits refresh LRU recency.
     *
     * @param {string | number} key
     * @returns {object | undefined}
     */
    getBindGroup(key) {
      if (key === null || key === undefined) {
        this._bindGroupMisses += 1;
        return undefined;
      }
      const cacheKey = String(key);
      const entry = this._bindGroupCache.get(cacheKey);
      if (entry === undefined) {
        this._bindGroupMisses += 1;
        return undefined;
      }
      if (!this._isBindGroupEntryValid(entry)) {
        this._evictBindGroupEntry(cacheKey, entry, 'stale');
        this._bindGroupMisses += 1;
        return undefined;
      }
      if (this._bindGroupCache.size > 1) {
        this._bindGroupCache.delete(cacheKey);
        this._bindGroupCache.set(cacheKey, entry);
      }
      this._bindGroupHits += 1;
      return entry.bindGroup;
    }

    /**
     * Non-mutating predicate returning `true` iff `key` is cached and all its
     * resource generation stamps are still live.
     *
     * @param {string | number} key
     * @returns {boolean}
     */
    hasBindGroup(key) {
      if (key === null || key === undefined) {
        return false;
      }
      const entry = this._bindGroupCache.get(String(key));
      return entry !== undefined && this._isBindGroupEntryValid(entry);
    }

    /**
     * Non-mutating inspection helper returning the stamped `(kind, handle, generation)`
     * tuples and current validity for a cached bind group entry.
     *
     * @param {string | number} key
     * @returns {{ key: string, stamps: Array<{ kind: string, handle: number, generation: number }>, valid: boolean } | undefined}
     */
    getBindGroupMeta(key) {
      if (key === null || key === undefined) {
        return undefined;
      }
      const cacheKey = String(key);
      const entry = this._bindGroupCache.get(cacheKey);
      if (entry === undefined) {
        return undefined;
      }
      return {
        key: cacheKey,
        stamps: entry.stamps.map((s) => ({
          kind: s.kind,
          handle: s.handle,
          generation: s.generation,
        })),
        valid: this._isBindGroupEntryValid(entry),
      };
    }

    deleteBindGroup(key) {
      if (key === null || key === undefined) {
        return false;
      }
      const cacheKey = String(key);
      const entry = this._bindGroupCache.get(cacheKey);
      if (entry === undefined) {
        return false;
      }
      return this._evictBindGroupEntry(cacheKey, entry, 'explicit');
    }

    clearBindGroups() {
      const count = this._bindGroupCache.size;
      if (this._fastBgSlots !== null) {
        for (let i = 0; i < this._fastBgSlots.length; i += 1) {
          const slot = this._fastBgSlots[i];
          if (slot !== null && slot.occupied) {
            if (slot.cacheEntry !== null) {
              slot.cacheEntry.evicted = true;
              slot.cacheEntry.fastSlotIndex = -1;
            }
            slot.occupied = false;
            slot.cacheEntry = null;
            slot.cacheKey = null;
          }
        }
      }
      this._bindGroupCache.clear();
      this._bindGroupResourceRefs.clear();
      return count;
    }

    /**
     * Sweep cached bind groups and evict any entries whose resource generations
     * advanced or whose underlying GPU objects were destroyed externally.
     *
     * @returns {number} Number of stale entries evicted.
     */
    evictStaleBindGroups() {
      let evicted = 0;
      for (const [cacheKey, entry] of this._bindGroupCache) {
        if (!this._isBindGroupEntryValid(entry)) {
          if (this._evictBindGroupEntry(cacheKey, entry, 'stale')) {
            evicted += 1;
          }
        }
      }
      return evicted;
    }

    _resolveBindGroupEntries(rawEntries, rawSlots) {
      let sorted;
      if (Array.isArray(rawEntries)) {
        let needsSort = false;
        let prevBinding = -Infinity;
        for (let i = 0; i < rawEntries.length; i += 1) {
          const item = rawEntries[i];
          const b = item && typeof item === 'object' ? Number(item.binding) || 0 : 0;
          if (b < prevBinding) {
            needsSort = true;
            break;
          }
          prevBinding = b;
        }
        if (needsSort) {
          sorted = rawEntries.slice().sort((a, b) => ((a && a.binding) || 0) - ((b && b.binding) || 0));
        } else {
          sorted = rawEntries;
        }
      } else if (Array.isArray(rawSlots)) {
        const len = rawSlots.length;
        sorted = new Array(len);
        for (let i = 0; i < len; i += 1) {
          sorted[i] = { binding: i, slot: rawSlots[i] };
        }
      } else {
        sorted = EMPTY_VIEW_FORMATS;
      }
      const gpuEntries = [];
      const stamps = [];
      const seenKeys = new Set();
      const directResources = [];
      const slotIndices = [];
      const keyParts = [];

      for (let i = 0; i < sorted.length; i += 1) {
        const item = sorted[i];
        if (!item || typeof item !== 'object') {
          return { ok: false, reason: 'invalid_entry' };
        }
        const binding = Number(item.binding) || 0;

        // Mode 1: ResourceTable slot index.
        if (item.slot !== undefined) {
          const slotIdx = item.slot;
          const slotEntry = this._getSlotEntry(slotIdx);
          if (!slotEntry || !slotEntry.bound) {
            this._missingLookups += 1;
            return { ok: false, reason: 'missing_slot', id: slotIdx };
          }
          const bgEntry = this.getBindGroupEntry(binding, slotIdx);
          if (!bgEntry) {
            return { ok: false, reason: 'missing_slot_resource', id: slotIdx };
          }
          const resolvedObj =
            bgEntry.resource && bgEntry.resource.buffer ? bgEntry.resource.buffer : bgEntry.resource;
          if (!resolvedObj || resolvedObj.destroyed === true) {
            return { ok: false, reason: 'destroyed_resource', id: slotIdx };
          }
          slotIndices.push(slotIdx);
          directResources.push(resolvedObj);
          if (slotEntry.handle !== null) {
            const bucket = this._normalizeGenerationBucket(slotEntry.kind);
            if (bucket !== null) {
              const liveGen = this._liveResourceGenerations[bucket].get(slotEntry.handle);
              if (liveGen === undefined) {
                return { ok: false, reason: 'destroyed_resource', id: slotEntry.handle };
              }
              this._appendStamp(stamps, seenKeys, bucket, slotEntry.handle, liveGen, directResources);
              if (bucket === 'texture') {
                const viewGen = this._liveResourceGenerations.textureView.get(slotEntry.handle);
                if (viewGen !== undefined) {
                  this._appendStamp(stamps, seenKeys, 'textureView', slotEntry.handle, viewGen, directResources);
                }
              }
            }
          } else if (this._objectOrigin !== null) {
            const origin = this._objectOrigin.get(resolvedObj);
            if (origin !== undefined) {
              if (!this.isResourceGenerationCurrent(origin.kind, origin.handle, origin.generation)) {
                return { ok: false, reason: 'stale_resource', id: origin.handle };
              }
              this._appendStamp(stamps, seenKeys, origin.kind, origin.handle, origin.generation, null);
            }
          }
          gpuEntries.push(bgEntry);
          keyParts.push(
            `${binding}:slot:${slotIdx}:${slotEntry.kind}:${slotEntry.handle !== null ? slotEntry.handle : this._getExternalObjectId(resolvedObj)}:${slotEntry.offset}:${slotEntry.size !== undefined ? slotEntry.size : -1}`
          );
          continue;
        }

        // Mode 2: ResourceTable handle descriptor ({ binding, kind, handle, offset?, size? }).
        if (item.handle !== undefined || item.id !== undefined || item.resourceId !== undefined) {
          const handle =
            item.handle !== undefined
              ? item.handle
              : item.id !== undefined
                ? item.id
                : item.resourceId;
          const bucket = this._normalizeGenerationBucket(item.kind);
          if (bucket === null || typeof handle !== 'number') {
            return { ok: false, reason: 'invalid_handle_entry', id: handle };
          }
          const liveGen = this._liveResourceGenerations[bucket].get(handle);
          if (liveGen === undefined) {
            this._missingLookups += 1;
            return { ok: false, reason: 'missing_resource', id: handle };
          }

          if (bucket === 'buffer') {
            const buf = this.buffers.get(handle);
            if (!buf || buf.destroyed === true) {
              this.invalidateBindGroupsForResource('buffer', handle);
              this._missingLookups += 1;
              return { ok: false, reason: 'destroyed_buffer', id: handle };
            }
            const offset =
              item.offset !== undefined && Number.isFinite(Number(item.offset))
                ? Math.max(0, Math.floor(Number(item.offset)))
                : 0;
            const size =
              item.size !== undefined && Number.isFinite(Number(item.size))
                ? Math.max(0, Math.floor(Number(item.size)))
                : undefined;
            const bufBinding = { buffer: buf, offset };
            if (size !== undefined) {
              bufBinding.size = size;
            }
            gpuEntries.push({ binding, resource: bufBinding });
            this._appendStamp(stamps, seenKeys, 'buffer', handle, liveGen, directResources);
            keyParts.push(`${binding}:buffer:${handle}:${offset}:${size !== undefined ? size : -1}`);
            continue;
          }

          if (bucket === 'texture' || bucket === 'textureView') {
            const view = this.textureViews.get(handle);
            const parentTexId =
              bucket === 'texture' ? handle : this._textureViewParents.get(handle);
            const parentTex = parentTexId !== undefined ? this.textures.get(parentTexId) : undefined;
            if (!view || view.destroyed === true || (parentTex && parentTex.destroyed === true)) {
              if (parentTexId !== undefined) {
                this.invalidateBindGroupsForResource('texture', parentTexId);
              } else {
                this.invalidateBindGroupsForResource(bucket, handle);
              }
              this._missingLookups += 1;
              return { ok: false, reason: 'destroyed_texture', id: handle };
            }
            if (parentTexId !== undefined && !this._liveResourceGenerations.texture.has(parentTexId)) {
              this.invalidateBindGroupsForResource('texture', parentTexId);
              this._missingLookups += 1;
              return { ok: false, reason: 'destroyed_texture', id: parentTexId };
            }
            gpuEntries.push({ binding, resource: view });
            this._appendStamp(stamps, seenKeys, bucket, handle, liveGen, directResources);
            if (bucket === 'texture') {
              const viewGen = this._liveResourceGenerations.textureView.get(handle);
              if (viewGen !== undefined) {
                this._appendStamp(stamps, seenKeys, 'textureView', handle, viewGen, directResources);
              }
            }
            keyParts.push(`${binding}:${bucket}:${handle}`);
            continue;
          }

          if (bucket === 'sampler') {
            const sampler = this.samplers.get(handle);
            if (!sampler || sampler.destroyed === true) {
              this.invalidateBindGroupsForResource('sampler', handle);
              this._missingLookups += 1;
              return { ok: false, reason: 'destroyed_sampler', id: handle };
            }
            gpuEntries.push({ binding, resource: sampler });
            this._appendStamp(stamps, seenKeys, 'sampler', handle, liveGen, directResources);
            keyParts.push(`${binding}:sampler:${handle}`);
            continue;
          }
        }

        // Mode 3: Direct WebGPU GPUBindGroupEntry ({ binding, resource }).
        const resource = item.resource;
        if (!resource || typeof resource !== 'object') {
          return { ok: false, reason: 'missing_direct_resource' };
        }
        const isBufferBinding = Boolean(resource.buffer && typeof resource.buffer === 'object');
        const targetObj = isBufferBinding ? resource.buffer : resource;
        let origin = this._objectOrigin !== null ? this._objectOrigin.get(targetObj) : undefined;
        if (targetObj.destroyed === true || resource.destroyed === true) {
          if (origin !== undefined) {
            this.invalidateBindGroupsForResource(origin.kind, origin.handle);
          } else {
            this.evictStaleBindGroups();
          }
          return { ok: false, reason: 'destroyed_direct_resource' };
        }

        if (origin !== undefined) {
          if (!this.isResourceGenerationCurrent(origin.kind, origin.handle, origin.generation)) {
            this.invalidateBindGroupsForResource(origin.kind, origin.handle);
            return { ok: false, reason: 'stale_direct_resource', id: origin.handle };
          }
          if (origin.parentTextureHandle !== undefined) {
            if (
              !this.isResourceGenerationCurrent(
                'texture',
                origin.parentTextureHandle,
                origin.parentTextureGeneration
              )
            ) {
              this.invalidateBindGroupsForResource('texture', origin.parentTextureHandle);
              return { ok: false, reason: 'stale_direct_texture', id: origin.parentTextureHandle };
            }
            const parentTex = this.textures.get(origin.parentTextureHandle);
            if (parentTex && parentTex.destroyed === true) {
              this.invalidateBindGroupsForResource('texture', origin.parentTextureHandle);
              return { ok: false, reason: 'destroyed_direct_texture', id: origin.parentTextureHandle };
            }
            if (parentTex) {
              directResources.push(parentTex);
            }
          }
          this._appendStamp(stamps, seenKeys, origin.kind, origin.handle, origin.generation, null);
          if (origin.parentTextureHandle !== undefined) {
            this._appendStamp(
              stamps,
              seenKeys,
              'texture',
              origin.parentTextureHandle,
              origin.parentTextureGeneration,
              null
            );
          }
          directResources.push(targetObj);
          if (isBufferBinding) {
            const offset = resource.offset !== undefined ? Math.max(0, Math.floor(Number(resource.offset))) : 0;
            const size = resource.size !== undefined ? Math.max(0, Math.floor(Number(resource.size))) : -1;
            keyParts.push(`${binding}:${origin.kind}:${origin.handle}:${offset}:${size}`);
          } else {
            keyParts.push(`${binding}:${origin.kind}:${origin.handle}`);
          }
        } else {
          directResources.push(targetObj);
          const extId = this._getExternalObjectId(targetObj);
          if (isBufferBinding) {
            const offset = resource.offset !== undefined ? Math.max(0, Math.floor(Number(resource.offset))) : 0;
            const size = resource.size !== undefined ? Math.max(0, Math.floor(Number(resource.size))) : -1;
            keyParts.push(`${binding}:ext:${extId}:${offset}:${size}`);
          } else {
            keyParts.push(`${binding}:ext:${extId}`);
          }
        }
        gpuEntries.push({ binding, resource });
      }

      const slotSnapshot = slotIndices.length > 0 ? this.captureSlotVersions(slotIndices) : null;
      return {
        ok: true,
        gpuEntries,
        stamps,
        directResources,
        slotSnapshot,
        entriesKey: keyParts.join('|'),
      };
    }

    _extractFastBindGroupProbe(entries, slots) {
      const isSlotsOnly = entries === null && Array.isArray(slots);
      const list = isSlotsOnly ? slots : Array.isArray(entries) ? entries : null;
      if (list === null) {
        return -1;
      }
      const count = list.length;
      if (count > this._probeMaxEntries) {
        return -1;
      }

      const bindings = this._probeBindings;
      const kinds = this._probeKinds;
      const handles = this._probeHandles;
      const generations = this._probeGenerations;
      const offsets = this._probeOffsets;
      const sizes = this._probeSizes;

      let needsSort = false;
      let prevBinding = -1;

      for (let i = 0; i < count; i += 1) {
        let binding;
        if (isSlotsOnly) {
          binding = i;
          const slotIdx = list[i];
          if (!isValidSlotIndex(slotIdx) || slotIdx > 0xffffffff) {
            return -1;
          }
          const slotEntry = this._getSlotEntry(slotIdx);
          if (!slotEntry || !slotEntry.bound || slotEntry.version > 0xffffffff) {
            return -1;
          }
          if (!this._isSlotEntryLiveForFastProbe(slotEntry)) {
            return -1;
          }
          bindings[i] = binding;
          kinds[i] = BG_KIND_SLOT;
          handles[i] = slotIdx;
          generations[i] = slotEntry.version;
          offsets[i] = slotEntry.offset;
          sizes[i] = slotEntry.size !== undefined ? slotEntry.size : -1;
        } else {
          const item = list[i];
          if (!item || typeof item !== 'object') {
            return -1;
          }
          const rawBinding = item.binding;
          binding = rawBinding === undefined ? 0 : Number(rawBinding) || 0;
          if (!Number.isInteger(binding) || binding < 0 || binding > 0xffffffff) {
            return -1;
          }

          if (item.slot !== undefined) {
            const slotIdx = item.slot;
            if (!isValidSlotIndex(slotIdx) || slotIdx > 0xffffffff) {
              return -1;
            }
            const slotEntry = this._getSlotEntry(slotIdx);
            if (!slotEntry || !slotEntry.bound || slotEntry.version > 0xffffffff) {
              return -1;
            }
            if (!this._isSlotEntryLiveForFastProbe(slotEntry)) {
              return -1;
            }
            bindings[i] = binding;
            kinds[i] = BG_KIND_SLOT;
            handles[i] = slotIdx;
            generations[i] = slotEntry.version;
            offsets[i] = slotEntry.offset;
            sizes[i] = slotEntry.size !== undefined ? slotEntry.size : -1;
          } else if (
            item.handle !== undefined ||
            item.id !== undefined ||
            item.resourceId !== undefined
          ) {
            const handle =
              item.handle !== undefined
                ? item.handle
                : item.id !== undefined
                  ? item.id
                  : item.resourceId;
            const bucket = this._normalizeGenerationBucketFast(item.kind);
            if (
              bucket === null ||
              typeof handle !== 'number' ||
              !Number.isInteger(handle) ||
              handle < 0 ||
              handle > 0xffffffff
            ) {
              return -1;
            }
            const liveGen = this._liveResourceGenerations[bucket].get(handle);
            if (liveGen === undefined || liveGen > 0xffffffff) {
              return -1;
            }

            if (bucket === 'buffer') {
              const buf = this.buffers.get(handle);
              if (!buf || buf.destroyed === true) {
                return -1;
              }
              const rawOff = item.offset;
              const offset =
                rawOff !== undefined && Number.isFinite(Number(rawOff))
                  ? Math.max(0, Math.floor(Number(rawOff)))
                  : 0;
              const rawSz = item.size;
              const size =
                rawSz !== undefined && Number.isFinite(Number(rawSz))
                  ? Math.max(0, Math.floor(Number(rawSz)))
                  : -1;
              bindings[i] = binding;
              kinds[i] = BG_KIND_BUFFER;
              handles[i] = handle;
              generations[i] = liveGen;
              offsets[i] = offset;
              sizes[i] = size;
            } else if (bucket === 'texture') {
              const view = this.textureViews.get(handle);
              const tex = this.textures.get(handle);
              if (!view || view.destroyed === true || (tex && tex.destroyed === true)) {
                return -1;
              }
              const viewGen = this._liveResourceGenerations.textureView.get(handle);
              const effGen = viewGen !== undefined ? viewGen : liveGen;
              if (effGen > 0xffffffff) {
                return -1;
              }
              bindings[i] = binding;
              kinds[i] = BG_KIND_TEXTURE;
              handles[i] = handle;
              generations[i] = effGen;
              offsets[i] = 0;
              sizes[i] = -1;
            } else if (bucket === 'textureView') {
              const view = this.textureViews.get(handle);
              const parentTexId = this._textureViewParents.get(handle);
              const parentTex =
                parentTexId !== undefined ? this.textures.get(parentTexId) : undefined;
              if (!view || view.destroyed === true || (parentTex && parentTex.destroyed === true)) {
                return -1;
              }
              if (parentTexId !== undefined && !this._liveResourceGenerations.texture.has(parentTexId)) {
                return -1;
              }
              bindings[i] = binding;
              kinds[i] = BG_KIND_TEXTURE_VIEW;
              handles[i] = handle;
              generations[i] = liveGen;
              offsets[i] = 0;
              sizes[i] = -1;
            } else if (bucket === 'sampler') {
              const sampler = this.samplers.get(handle);
              if (!sampler || sampler.destroyed === true) {
                return -1;
              }
              bindings[i] = binding;
              kinds[i] = BG_KIND_SAMPLER;
              handles[i] = handle;
              generations[i] = liveGen;
              offsets[i] = 0;
              sizes[i] = -1;
            } else {
              return -1;
            }
          } else {
            const resource = item.resource;
            if (!resource || typeof resource !== 'object') {
              return -1;
            }
            const isBufferBinding = Boolean(resource.buffer && typeof resource.buffer === 'object');
            const targetObj = isBufferBinding ? resource.buffer : resource;
            if (targetObj.destroyed === true || resource.destroyed === true) {
              return -1;
            }
            const origin = this._objectOrigin !== null ? this._objectOrigin.get(targetObj) : undefined;
            if (origin !== undefined) {
              const liveMap = this._liveResourceGenerations[origin.kind];
              if (!liveMap || liveMap.get(origin.handle) !== origin.generation) {
                return -1;
              }
              if (
                origin.handle < 0 ||
                origin.handle > 0xffffffff ||
                origin.generation > 0xffffffff
              ) {
                return -1;
              }
              if (origin.parentTextureHandle !== undefined) {
                if (
                  this._liveResourceGenerations.texture.get(origin.parentTextureHandle) !==
                  origin.parentTextureGeneration
                ) {
                  return -1;
                }
                const parentTex = this.textures.get(origin.parentTextureHandle);
                if (parentTex && parentTex.destroyed === true) {
                  return -1;
                }
              }
              let kindCode = 0;
              if (origin.kind === 'buffer') kindCode = BG_KIND_BUFFER;
              else if (origin.kind === 'texture') kindCode = BG_KIND_TEXTURE;
              else if (origin.kind === 'textureView') kindCode = BG_KIND_TEXTURE_VIEW;
              else if (origin.kind === 'sampler') kindCode = BG_KIND_SAMPLER;
              else return -1;

              bindings[i] = binding;
              kinds[i] = kindCode;
              handles[i] = origin.handle;
              generations[i] = origin.generation;
              if (isBufferBinding) {
                const rawOff = resource.offset;
                offsets[i] =
                  rawOff !== undefined && Number.isFinite(Number(rawOff))
                    ? Math.max(0, Math.floor(Number(rawOff)))
                    : 0;
                const rawSz = resource.size;
                sizes[i] =
                  rawSz !== undefined && Number.isFinite(Number(rawSz))
                    ? Math.max(0, Math.floor(Number(rawSz)))
                    : -1;
              } else {
                offsets[i] = 0;
                sizes[i] = -1;
              }
            } else {
              const extId = this._getExternalObjectId(targetObj);
              if (extId <= 0 || extId > 0xffffffff) {
                return -1;
              }
              bindings[i] = binding;
              kinds[i] = isBufferBinding ? BG_KIND_EXT_BUFFER : BG_KIND_EXT_OBJECT;
              handles[i] = extId;
              generations[i] = 0;
              if (isBufferBinding) {
                const rawOff = resource.offset;
                offsets[i] =
                  rawOff !== undefined && Number.isFinite(Number(rawOff))
                    ? Math.max(0, Math.floor(Number(rawOff)))
                    : 0;
                const rawSz = resource.size;
                sizes[i] =
                  rawSz !== undefined && Number.isFinite(Number(rawSz))
                    ? Math.max(0, Math.floor(Number(rawSz)))
                    : -1;
              } else {
                offsets[i] = 0;
                sizes[i] = -1;
              }
            }
          }
        }

        if (binding < prevBinding) {
          needsSort = true;
        }
        prevBinding = binding;
      }

      if (needsSort) {
        for (let i = 1; i < count; i += 1) {
          const b = bindings[i];
          const k = kinds[i];
          const h = handles[i];
          const g = generations[i];
          const off = offsets[i];
          const sz = sizes[i];
          let j = i - 1;
          while (j >= 0 && bindings[j] > b) {
            bindings[j + 1] = bindings[j];
            kinds[j + 1] = kinds[j];
            handles[j + 1] = handles[j];
            generations[j + 1] = generations[j];
            offsets[j + 1] = offsets[j];
            sizes[j + 1] = sizes[j];
            j -= 1;
          }
          bindings[j + 1] = b;
          kinds[j + 1] = k;
          handles[j + 1] = h;
          generations[j + 1] = g;
          offsets[j + 1] = off;
          sizes[j + 1] = sz;
        }
      }

      return count;
    }

    _isSlotEntryLiveForFastProbe(slotEntry) {
      if (slotEntry.resource !== null) {
        if (slotEntry.resource.destroyed === true) {
          return false;
        }
        if (this._objectOrigin !== null) {
          const origin = this._objectOrigin.get(slotEntry.resource);
          if (origin !== undefined) {
            const liveMap = this._liveResourceGenerations[origin.kind];
            if (!liveMap || liveMap.get(origin.handle) !== origin.generation) {
              return false;
            }
            if (origin.parentTextureHandle !== undefined) {
              if (
                this._liveResourceGenerations.texture.get(origin.parentTextureHandle) !==
                origin.parentTextureGeneration
              ) {
                return false;
              }
              const parentTex = this.textures.get(origin.parentTextureHandle);
              if (parentTex && parentTex.destroyed === true) {
                return false;
              }
            }
          }
        }
        return true;
      }
      if (slotEntry.handle === null) {
        return false;
      }
      const bucket = this._normalizeGenerationBucketFast(slotEntry.kind);
      if (bucket === null) {
        return false;
      }
      if (!this._liveResourceGenerations[bucket].has(slotEntry.handle)) {
        return false;
      }
      if (bucket === 'buffer') {
        const buf = this.buffers.get(slotEntry.handle);
        return Boolean(buf && buf.destroyed !== true);
      }
      if (bucket === 'texture' || bucket === 'textureView') {
        const tex = this.textures.get(slotEntry.handle);
        const view = this.textureViews.get(slotEntry.handle);
        if ((tex && tex.destroyed === true) || (view && view.destroyed === true)) {
          return false;
        }
        if (slotEntry.cachedView && slotEntry.cachedView.destroyed === true) {
          return false;
        }
        return Boolean(view || tex);
      }
      if (bucket === 'sampler') {
        const sampler = this.samplers.get(slotEntry.handle);
        return Boolean(sampler && sampler.destroyed !== true);
      }
      return false;
    }

    _hashFastProbe(layoutId, entryCount) {
      const pair = this._probeHashPair;
      if (this._bindGroupHashFn !== null) {
        pair[0] = 0;
        pair[1] = 0;
        const customHash = this._bindGroupHashFn(
          layoutId,
          entryCount,
          this._probeBindings,
          this._probeHandles,
          this._probeGenerations,
          pair,
          this._probeKinds,
          this._probeOffsets,
          this._probeSizes
        );
        if (
          pair[0] === 0 &&
          pair[1] === 0 &&
          typeof customHash === 'number' &&
          Number.isFinite(customHash)
        ) {
          pair[0] = customHash >>> 0;
          pair[1] = customHash > 0xffffffff ? Math.floor(customHash / TWO_POW_32) >>> 0 : 0;
        }
        return;
      }
      computeBindGroupHashFast(
        layoutId,
        entryCount,
        this._probeBindings,
        this._probeHandles,
        this._probeGenerations,
        pair,
        this._probeKinds,
        this._probeOffsets,
        this._probeSizes
      );
    }

    _fastSlotMatchesProbe(slot, layoutId, entryCount) {
      if (slot.layoutId !== layoutId || slot.entryCount !== entryCount) {
        return false;
      }
      const bA = slot.bindings;
      const kA = slot.kinds;
      const hA = slot.handles;
      const gA = slot.generations;
      const oA = slot.offsets;
      const sA = slot.sizes;
      const bB = this._probeBindings;
      const kB = this._probeKinds;
      const hB = this._probeHandles;
      const gB = this._probeGenerations;
      const oB = this._probeOffsets;
      const sB = this._probeSizes;
      for (let i = 0; i < entryCount; i += 1) {
        if (
          bA[i] !== bB[i] ||
          hA[i] !== hB[i] ||
          gA[i] !== gB[i] ||
          kA[i] !== kB[i] ||
          oA[i] !== oB[i] ||
          sA[i] !== sB[i]
        ) {
          return false;
        }
      }
      return true;
    }

    _probeFastBindGroupCache(layoutId, entryCount) {
      if (this._fastBgSlots === null) {
        return null;
      }
      this._probeEpoch = (this._probeEpoch + 1) >>> 0;
      this._hashFastProbe(layoutId, entryCount);
      this._zeroAllocKeyProbes += 1;

      const hashLo = this._probeHashPair[0];
      const hashHi = this._probeHashPair[1];
      const mask = this._fastBgMask;
      const start = hashLo & mask;
      const slots = this._fastBgSlots;
      const probeLimit = Math.min(FAST_BIND_GROUP_PROBE_WINDOW, this._fastBgBucketCount);

      for (let p = 0; p < probeLimit; p += 1) {
        const idx = (start + p) & mask;
        const slot = slots[idx];
        if (slot === null || !slot.occupied) {
          continue;
        }
        if (slot.hashLo !== hashLo || slot.hashHi !== hashHi) {
          continue;
        }
        if (!this._fastSlotMatchesProbe(slot, layoutId, entryCount)) {
          this._hashCollisions += 1;
          continue;
        }
        const entry = slot.cacheEntry;
        if (entry === null || entry.evicted) {
          slot.occupied = false;
          slot.cacheEntry = null;
          slot.cacheKey = null;
          continue;
        }
        if (!this._isBindGroupEntryValid(entry)) {
          this._evictBindGroupEntry(slot.cacheKey, entry, 'stale');
          return null;
        }
        if (this._bindGroupCache.size > 1) {
          this._bindGroupCache.delete(entry.key);
          this._bindGroupCache.set(entry.key, entry);
        }
        this._zeroAllocKeyHits += 1;
        this._bindGroupHits += 1;
        return entry.bindGroup;
      }
      return null;
    }

    _insertFastBindGroupEntry(entry, layoutId, entryCount) {
      if (this._fastBgSlots === null || !entry || entry.evicted || entry.fastSlotIndex >= 0) {
        return;
      }
      const hashLo = this._probeHashPair[0];
      const hashHi = this._probeHashPair[1];
      const mask = this._fastBgMask;
      const start = hashLo & mask;
      const slots = this._fastBgSlots;
      const probeLimit = Math.min(FAST_BIND_GROUP_PROBE_WINDOW, this._fastBgBucketCount);

      let targetIdx = -1;
      for (let p = 0; p < probeLimit; p += 1) {
        const idx = (start + p) & mask;
        const slot = slots[idx];
        if (
          slot === null ||
          !slot.occupied ||
          slot.cacheEntry === null ||
          slot.cacheEntry.evicted
        ) {
          targetIdx = idx;
          break;
        }
      }

      if (targetIdx === -1) {
        const victimOffset = this._fastBgReplaceCursor % probeLimit;
        this._fastBgReplaceCursor = (this._fastBgReplaceCursor + 1) >>> 0;
        targetIdx = (start + victimOffset) & mask;
        const victimSlot = slots[targetIdx];
        if (victimSlot !== null && victimSlot.cacheEntry !== null) {
          victimSlot.cacheEntry.fastSlotIndex = -1;
        }
      }

      let slot = slots[targetIdx];
      const minCap = entryCount > 8 ? entryCount : 8;
      if (slot === null) {
        slot = {
          occupied: true,
          hashLo,
          hashHi,
          layoutId,
          entryCount,
          capacity: minCap,
          bindings: new Uint32Array(minCap),
          kinds: new Uint8Array(minCap),
          handles: new Uint32Array(minCap),
          generations: new Uint32Array(minCap),
          offsets: new Float64Array(minCap),
          sizes: new Float64Array(minCap),
          cacheKey: entry.key,
          cacheEntry: entry,
        };
        slots[targetIdx] = slot;
      } else {
        if (slot.capacity < entryCount) {
          slot.capacity = entryCount;
          slot.bindings = new Uint32Array(entryCount);
          slot.kinds = new Uint8Array(entryCount);
          slot.handles = new Uint32Array(entryCount);
          slot.generations = new Uint32Array(entryCount);
          slot.offsets = new Float64Array(entryCount);
          slot.sizes = new Float64Array(entryCount);
        }
        slot.occupied = true;
        slot.hashLo = hashLo;
        slot.hashHi = hashHi;
        slot.layoutId = layoutId;
        slot.entryCount = entryCount;
        slot.cacheKey = entry.key;
        slot.cacheEntry = entry;
      }

      const bSrc = this._probeBindings;
      const kSrc = this._probeKinds;
      const hSrc = this._probeHandles;
      const gSrc = this._probeGenerations;
      const oSrc = this._probeOffsets;
      const sSrc = this._probeSizes;
      for (let i = 0; i < entryCount; i += 1) {
        slot.bindings[i] = bSrc[i];
        slot.kinds[i] = kSrc[i];
        slot.handles[i] = hSrc[i];
        slot.generations[i] = gSrc[i];
        slot.offsets[i] = oSrc[i];
        slot.sizes[i] = sSrc[i];
      }
      entry.fastSlotIndex = targetIdx;
    }

    /**
     * Get a cached `GPUBindGroup` or create, stamp, and cache a new one under
     * the bounded LRU policy (`maxBindGroups`).
     *
     * Supports:
     * - `getOrCreateBindGroup(key, descriptor)` (`{ layout, entries?, slots?, label?, resources?, factory? }`)
     * - `getOrCreateBindGroup(key, factory, resourceRefs)`
     * - `getOrCreateBindGroup(key, layout, entries, options)`
     * - `getOrCreateBindGroup(layout, entries, options)`
     * - `getOrCreateBindGroup(descriptor)` (auto-derives deterministic key if `descriptor.key` is omitted)
     *
     * Returns `null` (without calling `device.createBindGroup`) if any referenced
     * handle is missing or destroyed.
     *
     * @returns {object | null}
     */
    getOrCreateBindGroup(arg0, arg1, arg2, arg3) {
      let explicitKey = null;
      let layout = null;
      let entries = null;
      let slots = null;
      let label;
      let factory = null;
      let explicitResourceRefs = null;

      if (typeof arg0 === 'string' || typeof arg0 === 'number') {
        explicitKey = String(arg0);
        if (typeof arg1 === 'function') {
          factory = arg1;
          explicitResourceRefs = Array.isArray(arg2)
            ? arg2
            : arg2 && Array.isArray(arg2.resources)
              ? arg2.resources
              : [];
        } else if (arg1 && typeof arg1 === 'object' && Array.isArray(arg2)) {
          layout = arg1;
          entries = arg2;
          if (arg3 && typeof arg3 === 'object') {
            label = arg3.label;
            explicitResourceRefs = arg3.resources || null;
            if (typeof arg3.factory === 'function') factory = arg3.factory;
          }
        } else if (arg1 && typeof arg1 === 'object') {
          layout = arg1.layout || null;
          entries = arg1.entries || null;
          slots = arg1.slots || null;
          label = arg1.label;
          explicitResourceRefs = arg1.resources || null;
          factory =
            typeof arg1.factory === 'function'
              ? arg1.factory
              : typeof arg1.create === 'function'
                ? arg1.create
                : null;
        }
      } else if (arg0 && typeof arg0 === 'object') {
        if (Array.isArray(arg1)) {
          layout = arg0;
          entries = arg1;
          const opts = arg2 && typeof arg2 === 'object' ? arg2 : null;
          if (opts) {
            if (opts.key !== undefined && opts.key !== null) {
              explicitKey = String(opts.key);
            }
            label = opts.label;
            explicitResourceRefs = opts.resources || null;
            if (typeof opts.factory === 'function') factory = opts.factory;
          }
        } else {
          if (arg0.key !== undefined && arg0.key !== null) {
            explicitKey = String(arg0.key);
          }
          layout = arg0.layout || null;
          entries = arg0.entries || null;
          slots = arg0.slots || null;
          label = arg0.label;
          explicitResourceRefs = arg0.resources || null;
          factory =
            typeof arg0.factory === 'function'
              ? arg0.factory
              : typeof arg0.create === 'function'
                ? arg0.create
                : null;
        }
      } else {
        throw new TypeError('ResourceTable getOrCreateBindGroup requires a key, layout, or descriptor');
      }

      // Fast-path lookup when an explicit cache key was provided.
      if (explicitKey !== null) {
        const existing = this._bindGroupCache.get(explicitKey);
        if (existing !== undefined) {
          if (this._isBindGroupEntryValid(existing)) {
            if (this._bindGroupCache.size > 1) {
              this._bindGroupCache.delete(explicitKey);
              this._bindGroupCache.set(explicitKey, existing);
            }
            this._bindGroupHits += 1;
            return existing.bindGroup;
          }
          this._evictBindGroupEntry(explicitKey, existing, 'stale');
        }
      }

      // Zero-alloc numeric 64-bit fast-path probe when auto-keying entries/slots (W320).
      let fastProbeCount = -1;
      let fastProbeLayoutId = 0;
      let fastProbeEpoch = 0;
      if (
        explicitKey === null &&
        this._fastBgSlots !== null &&
        (entries !== null || slots !== null) &&
        (!Array.isArray(explicitResourceRefs) || explicitResourceRefs.length === 0)
      ) {
        fastProbeCount = this._extractFastBindGroupProbe(entries, slots);
        if (fastProbeCount >= 0) {
          fastProbeLayoutId = this._getNumericLayoutId(layout);
          const fastHit = this._probeFastBindGroupCache(fastProbeLayoutId, fastProbeCount);
          if (fastHit !== null) {
            return fastHit;
          }
          fastProbeEpoch = this._probeEpoch;
        }
      }

      let resolved = null;
      let cacheKey = explicitKey;

      if (entries !== null || slots !== null) {
        resolved = this._resolveBindGroupEntries(entries, slots);
        if (!resolved.ok) {
          this._bindGroupMisses += 1;
          this._warn(
            'bind_group_invalid_resource',
            typeof resolved.id === 'number' ? resolved.id : -1,
            `skipped bind group creation: ${resolved.reason}`
          );
          return null;
        }
        if (cacheKey === null) {
          cacheKey = `L:${this._getLayoutId(layout)}|${resolved.entriesKey}`;
          const existing = this._bindGroupCache.get(cacheKey);
          if (existing !== undefined) {
            if (this._isBindGroupEntryValid(existing)) {
              if (this._bindGroupCache.size > 1) {
                this._bindGroupCache.delete(cacheKey);
                this._bindGroupCache.set(cacheKey, existing);
              }
              if (fastProbeCount >= 0 && this._probeEpoch === fastProbeEpoch) {
                this._insertFastBindGroupEntry(existing, fastProbeLayoutId, fastProbeCount);
              }
              this._bindGroupHits += 1;
              return existing.bindGroup;
            }
            this._evictBindGroupEntry(cacheKey, existing, 'stale');
          }
        }
      }

      if (cacheKey === null) {
        throw new TypeError(
          'ResourceTable getOrCreateBindGroup requires an explicit key or layout + entries/slots'
        );
      }

      this._bindGroupMisses += 1;

      const stamps = resolved ? resolved.stamps : [];
      const seenKeys = new Set();
      for (let i = 0; i < stamps.length; i += 1) {
        seenKeys.add(this._resourceRefKey(stamps[i].kind, stamps[i].handle));
      }
      const directResources = resolved ? resolved.directResources : [];
      let slotSnapshot = resolved ? resolved.slotSnapshot : null;

      if (Array.isArray(explicitResourceRefs) && explicitResourceRefs.length > 0) {
        const extra = this._normalizeBindGroupStamps(explicitResourceRefs, true);
        if (!extra.ok) {
          this._warn(
            'bind_group_invalid_resource',
            -1,
            'skipped bind group creation: referenced resource handle is missing or stale'
          );
          return null;
        }
        for (let i = 0; i < extra.stamps.length; i += 1) {
          const s = extra.stamps[i];
          const rk = this._resourceRefKey(s.kind, s.handle);
          if (!seenKeys.has(rk)) {
            seenKeys.add(rk);
            stamps.push(s);
          }
        }
        for (let i = 0; i < extra.directResources.length; i += 1) {
          directResources.push(extra.directResources[i]);
        }
        if (slotSnapshot === null && extra.slotSnapshot !== null) {
          slotSnapshot = extra.slotSnapshot;
        }
      }

      let bindGroup;
      if (factory !== null) {
        bindGroup = factory(resolved ? resolved.gpuEntries : undefined);
      } else {
        if (!layout) {
          throw new TypeError(
            'ResourceTable getOrCreateBindGroup requires a layout or factory function'
          );
        }
        if (typeof this.device.createBindGroup !== 'function') {
          throw new TypeError(
            'ResourceTable device does not implement createBindGroup()'
          );
        }
        const bgDesc = {
          layout,
          entries: resolved ? resolved.gpuEntries : [],
        };
        if (typeof label === 'string' && label.length > 0) {
          bgDesc.label = label;
        }
        bindGroup = this.device.createBindGroup(bgDesc);
      }

      if (!bindGroup || typeof bindGroup !== 'object') {
        return null;
      }

      const storedEntry = this._storeBindGroupEntryRecord(
        cacheKey,
        bindGroup,
        stamps,
        directResources,
        slotSnapshot
      );
      if (storedEntry !== null && fastProbeCount >= 0 && this._probeEpoch === fastProbeEpoch) {
        this._insertFastBindGroupEntry(storedEntry, fastProbeLayoutId, fastProbeCount);
      }
      return bindGroup;
    }

    destroyAll() {
      const prevSuppressDefer = this._suppressDeferDestroys;
      this._suppressDeferDestroys = true;
      this._inFrame = false;
      try {
        for (const id of Array.from(this.buffers.keys())) {
          this.destroyBuffer(id);
        }
        for (const id of Array.from(this.textures.keys())) {
          this.destroyTexture(id);
        }
        for (const id of Array.from(this.textureViews.keys())) {
          this.destroyTextureView(id);
        }
        for (const id of Array.from(this.samplers.keys())) {
          this.destroy(id);
        }
        for (const id of Array.from(this.shaderModules.keys())) {
          this.destroy(id);
        }
        this.flushDeferredDestroys();
        this._writeScratch = null;
        this._liveBufferBytes = 0;
        this._liveTextureBytes = 0;
      } finally {
        this._suppressDeferDestroys = prevSuppressDefer;
      }
    }

    /**
     * Destroy all tracked GPU resources, clear bound descriptor slots, clear
     * cached bind groups, and clear internal handle/metadata maps. Monotonic
     * slot/table versions and resource generation counters are preserved so
     * downstream bind-group caches never observe stale version reuse.
     * Pass `{ resetStats: true }` to also reset telemetry counters.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this.clearSlots();
      this.destroyAll();
      this.clearBindGroups();
      this._bufferMeta.clear();
      this.textureViews.clear();
      this._textureViewParents.clear();
      this._textureChildViews.clear();
      this._textureMeta.clear();
      if (options && options.resetStats) {
        this.resetStats();
      }
    }

    /**
     * Return sanitized diagnostic counters (no host paths or raw payload data).
     */
    getStats() {
      const totalBindGroupLookups = this._bindGroupHits + this._bindGroupMisses;
      const rawHitRate =
        totalBindGroupLookups > 0 ? this._bindGroupHits / totalBindGroupLookups : 0;
      const bindGroupHitRate = Math.min(1, Math.max(0, rawHitRate));

      const pendingDeferredDestroys = this._getPendingDeferredDestroyCount();
      const writeScratchBytes =
        this._writeScratch && typeof this._writeScratch.byteLength === 'number'
          ? this._writeScratch.byteLength
          : 0;

      return {
        bcFormatFallbackCount: this._bcFormatFallbackCount,
        bcDimensionAlignCount: this._bcDimensionAlignCount,
        bufferAlignPadCount: this._bufferAlignPadCount,
        mipClampCount: this._mipClampCount,
        sampleClampCount: this._sampleClampCount,
        buffersCreated: this._buffersCreated,
        texturesCreated: this._texturesCreated,
        texture2Created: this._texture2Created,
        samplersCreated: this._samplersCreated,
        shaderModulesCreated: this._shaderModulesCreated,
        bufferWrites: this._bufferWrites,
        buffersActive: this.buffers.size,
        texturesActive: this.textures.size,
        textureViewsActive: this.textureViews.size,
        samplersActive: this.samplers.size,
        shaderModulesActive: this.shaderModules.size,
        liveBufferBytes: this._liveBufferBytes,
        peakBufferBytes: this._peakBufferBytes,
        liveTextureBytes: this._liveTextureBytes,
        peakTextureBytes: this._peakTextureBytes,
        writeScratchBytes,
        destroyCount: this._destroyCount,
        destroyedBuffers: this._destroyedBuffers,
        destroyedTextures: this._destroyedTextures,
        chunkedWrites: this._chunkedWrites,
        maxWriteChunkBytes: this._maxWriteChunkBytes,
        deferredDestroyCount: this._deferredDestroyCount,
        pendingDeferredDestroyCount: pendingDeferredDestroys,
        pendingDeferredDestroys,
        pendingDeferredDestroyBytes: this._pendingDeferredDestroyBytes,
        viewDimensionGuardCount: this._viewDimensionGuardCount,
        promoted1DTo2DCount: this._promoted1DTo2DCount,
        errors: this._errors,
        liveHandles: this._getLiveHandleCount(),
        peakLiveHandles: this._peakLiveHandles,
        overwrites: this._overwrites,
        missingLookups: this._missingLookups,
        totalInserts: this._totalInserts,
        totalDeletes: this._totalDeletes,
        slotWrites: this._slotWrites,
        elidedNoOpWrites: this._elidedNoOpWrites,
        slotLookups: this._slotLookups,
        tableVersion: this._tableVersion,
        activeSlots: this._activeSlots,
        bindGroupHits: this._bindGroupHits,
        bindGroupMisses: this._bindGroupMisses,
        bindGroupHitRate,
        staleBindGroupEvictions: this._staleBindGroupEvictions,
        lruBindGroupEvictions: this._lruBindGroupEvictions,
        cachedBindGroups: this._bindGroupCache.size,
        maxBindGroups: this._maxBindGroups,
        zeroAllocKeyProbes: this._zeroAllocKeyProbes,
        zeroAllocKeyHits: this._zeroAllocKeyHits,
        hashCollisions: this._hashCollisions,
      };
    }

    resetStats() {
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;
      this._buffersCreated = 0;
      this._texturesCreated = 0;
      this._texture2Created = 0;
      this._samplersCreated = 0;
      this._shaderModulesCreated = 0;
      this._bufferWrites = 0;
      this._destroyCount = 0;
      this._destroyedBuffers = 0;
      this._destroyedTextures = 0;
      this._chunkedWrites = 0;
      this._deferredDestroyCount = 0;
      this._viewDimensionGuardCount = 0;
      this._promoted1DTo2DCount = 0;
      this._errors = 0;
      this._lastError = null;
      this._peakLiveHandles = this._getLiveHandleCount();
      this._peakBufferBytes = this._liveBufferBytes;
      this._peakTextureBytes = this._liveTextureBytes;
      this._overwrites = 0;
      this._missingLookups = 0;
      this._totalInserts = 0;
      this._totalDeletes = 0;
      this._slotWrites = 0;
      this._elidedNoOpWrites = 0;
      this._slotLookups = 0;
      this._bindGroupHits = 0;
      this._bindGroupMisses = 0;
      this._staleBindGroupEvictions = 0;
      this._lruBindGroupEvictions = 0;
      this._zeroAllocKeyProbes = 0;
      this._zeroAllocKeyHits = 0;
      this._hashCollisions = 0;
    }
  }

  ResourceTable.isViewDimensionCompatible = isViewDimensionCompatible;
  ResourceTable.canonicalizeTextureViewDescriptorKey = canonicalizeTextureViewDescriptorKey;
  ResourceTable.canonicalizeSlotDescriptorKey = canonicalizeSlotDescriptorKey;

  const exportsObj = Object.freeze({
    CREATE_TEXTURE2_OPCODE,
    CREATE_TEXTURE2_PAYLOAD_BYTES,
    MAX_TEXTURE_VIEW_FORMATS,
    GPU_BUFFER_USAGE,
    GPU_TEXTURE_USAGE,
    V1_TEXTURE_FORMAT_NAMES,
    TEXTURE_FORMAT_NAMES,
    TEXTURE_DIMENSION_NAMES,
    BC_TEXTURE_FORMATS,
    BC_FALLBACK_FORMAT,
    TEXTURE_COMPRESSION_BC_FEATURE,
    BUFFER_ALIGNMENT_BYTES,
    UNIFORM_BUFFER_MIN_BYTES,
    DEFAULT_MAX_WRITE_CHUNK_BYTES,
    DEFAULT_FAST_SLOT_CAPACITY,
    DEFAULT_MAX_BIND_GROUPS,
    DEFAULT_FAST_BIND_GROUP_BUCKETS,
    isBcFormat,
    estimateTextureByteSize,
    isViewDimensionCompatible,
    deviceSupportsBc,
    computeMaxMipLevels,
    clampMipLevelCount,
    clampSampleCount,
    alignBcDimension,
    alignBufferSize,
    padBufferUploadData,
    decodeCreateTexture2,
    sanitizeTextureDescriptor,
    computeBindGroupHashFast,
    canonicalizeTextureViewDescriptorKey,
    canonicalizeSlotDescriptorKey,
    ResourceTable,
    GpuResourceTable: ResourceTable,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ResourceTable = exportsObj;
  } else if (globalScope) {
    globalScope.PlayGta6ResourceTable = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      CREATE_TEXTURE2_OPCODE,
      CREATE_TEXTURE2_PAYLOAD_BYTES,
      MAX_TEXTURE_VIEW_FORMATS,
      GPU_BUFFER_USAGE,
      GPU_TEXTURE_USAGE,
      V1_TEXTURE_FORMAT_NAMES,
      TEXTURE_FORMAT_NAMES,
      TEXTURE_DIMENSION_NAMES,
      BC_TEXTURE_FORMATS,
      BC_FALLBACK_FORMAT,
      TEXTURE_COMPRESSION_BC_FEATURE,
      BUFFER_ALIGNMENT_BYTES,
      UNIFORM_BUFFER_MIN_BYTES,
      DEFAULT_MAX_WRITE_CHUNK_BYTES,
      DEFAULT_FAST_SLOT_CAPACITY,
      DEFAULT_MAX_BIND_GROUPS,
      DEFAULT_FAST_BIND_GROUP_BUCKETS,
      isBcFormat,
      estimateTextureByteSize,
      isViewDimensionCompatible,
      deviceSupportsBc,
      computeMaxMipLevels,
      clampMipLevelCount,
      clampSampleCount,
      alignBcDimension,
      alignBufferSize,
      padBufferUploadData,
      decodeCreateTexture2,
      sanitizeTextureDescriptor,
      computeBindGroupHashFast,
      canonicalizeTextureViewDescriptorKey,
      canonicalizeSlotDescriptorKey,
      ResourceTable,
      GpuResourceTable: ResourceTable,
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
