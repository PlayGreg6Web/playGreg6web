'use strict';

// WebGPU missing-shader fallback router + Frame-1 telemetry (Track 4/3 bridge S12).
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6ShaderFallbackRouter)
// and Node.js (module.exports). Hash / stage conventions match web/workers/wgsl_pack_loader.js,
// web/workers/reflection_loader.js, and tools/gen_fallback_shaders.py.
(function (root) {
  /** WGPK / convert.rs numeric stage codes (Vertex=1, Fragment=2, Compute=3). */
  const SHADER_STAGE = Object.freeze({
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 3,
  });

  const STAGE_NAMES = Object.freeze({
    [SHADER_STAGE.VERTEX]: 'vertex',
    [SHADER_STAGE.FRAGMENT]: 'fragment',
    [SHADER_STAGE.COMPUTE]: 'compute',
  });

  const STAGE_NAME_TO_CODE = Object.freeze({
    vertex: SHADER_STAGE.VERTEX,
    vs: SHADER_STAGE.VERTEX,
    fragment: SHADER_STAGE.FRAGMENT,
    fs: SHADER_STAGE.FRAGMENT,
    pixel: SHADER_STAGE.FRAGMENT,
    ps: SHADER_STAGE.FRAGMENT,
    compute: SHADER_STAGE.COMPUTE,
    cs: SHADER_STAGE.COMPUTE,
  });

  const DEFAULT_ENTRY_POINTS = Object.freeze({
    vertex: 'vs_main',
    fragment: 'fs_main',
    compute: 'cs_main',
  });

  const U64_MAX = 0xffffffffffffffffn;
  const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
  const DEFAULT_MAX_TRACKED_MISSING = 512;
  const DEFAULT_MAX_MISSING_HASHES = 512;
  const DEFAULT_MAX_TRACKED_MISSING_HASHES = 2048;
  const DEFAULT_MAX_CATEGORIZED_MISSING_HASHES = 8192;
  const DEFAULT_MAX_RESOLVE_CACHE_ENTRIES = 8192;
  const DEFAULT_MAX_RESOLVE_CACHE_PACK_WGSL_BYTES = 8 * 1024 * 1024;
  const DEFAULT_MAX_REAL_SHADERS = 4096;
  const DEFAULT_MAX_COMPILE_FAILURE_ENTRIES = 2048;
  const DEFAULT_TOP_MISSING_LIMIT = 16;
  const DEFAULT_MISSING_SUMMARY_TOP_LIMIT = 32;
  const SAFE_REASON_TOKEN_REGEX = /^[a-z][a-z0-9_]{0,63}$/;

  const ENTRY_POINT_REGEXES = Object.freeze({
    vertex: /@vertex\b[\s\S]*?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/,
    fragment: /@fragment\b[\s\S]*?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/,
    compute: /@compute\b[\s\S]*?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/,
  });
  const FALLBACK_FN_REGEX = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/;

  /**
   * Canonical missing-shader failure categories for live Chrome triage telemetry (W179).
   */
  const MISSING_SHADER_REASONS = Object.freeze({
    NOT_IN_PACK: 'not_in_pack',
    REFLECTION_FAILED: 'reflection_failed',
    COMPILE_FAILED: 'compile_failed',
    OTHER: 'other',
  });

  const MISSING_SHADER_REASON_ORDER = Object.freeze([
    MISSING_SHADER_REASONS.NOT_IN_PACK,
    MISSING_SHADER_REASONS.REFLECTION_FAILED,
    MISSING_SHADER_REASONS.COMPILE_FAILED,
    MISSING_SHADER_REASONS.OTHER,
  ]);

  const MISSING_SHADER_REASON_ALIASES = Object.freeze({
    not_in_pack: MISSING_SHADER_REASONS.NOT_IN_PACK,
    missing: MISSING_SHADER_REASONS.NOT_IN_PACK,
    missing_from_pack: MISSING_SHADER_REASONS.NOT_IN_PACK,
    missing_pack_entry: MISSING_SHADER_REASONS.NOT_IN_PACK,
    pack_miss: MISSING_SHADER_REASONS.NOT_IN_PACK,
    reflection_failed: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    reflection_error: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    reflection_miss: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    no_reflection: MISSING_SHADER_REASONS.REFLECTION_FAILED,
    compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    compilation_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    compile_error: MISSING_SHADER_REASONS.COMPILE_FAILED,
    wgsl_compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    shader_compile_failed: MISSING_SHADER_REASONS.COMPILE_FAILED,
    other: MISSING_SHADER_REASONS.OTHER,
  });

  /**
   * Minimal original WebGPU WGSL fallbacks used when no pack (and no S9 fallback hash)
   * is available yet. Never derived from copyrighted game shaders.
   */
  const BUILTIN_FALLBACK_WGSL = Object.freeze({
    vertex: [
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    let p = positions[vertex_index];',
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
    compute: [
      '@compute @workgroup_size(1)',
      'fn cs_main() {}',
      '',
    ].join('\n'),
  });

  /**
   * Frame-1 D3D12 fullscreen-triangle + CBV fallback WGSL pair.
   * Matches `DrawInstanced(3, 1)` with a 1-CBV `group(0) binding(0)` descriptor table
   * (e.g. 256-byte `array<vec4<f32>, 16>` constant buffer) onto an RGBA8 render target.
   * Works with or without bound vertex attributes via `@builtin(vertex_index) vid: u32`,
   * uses bitcast-masked CBV modulation so output is never NaN and always `alpha = 1.0`.
   */
  const FULLSCREEN_TRIANGLE_CBV_WGSL = Object.freeze({
    vertex: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;',
      '',
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {',
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    var uvs = array<vec2<f32>, 3>(',
      '        vec2<f32>(0.0, 1.0),',
      '        vec2<f32>(2.0, 1.0),',
      '        vec2<f32>(0.0, -1.0),',
      '    );',
      '    let idx = vid % 3u;',
      '    let cbv_keep = f32(bitcast<u32>(frame_cbv.data[0].x) & 0xffu) * 0.0;',
      '    var out: VsOut;',
      '    out.position = vec4<f32>(positions[idx], cbv_keep, 1.0);',
      '    out.uv = uvs[idx];',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;',
      '',
      'struct FsIn {',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@fragment',
      'fn fs_main(input: FsIn) -> @location(0) vec4<f32> {',
      '    let uv = clamp(input.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));',
      '    let b0 = bitcast<u32>(frame_cbv.data[0].x);',
      '    let b1 = bitcast<u32>(frame_cbv.data[0].y);',
      '    let b2 = bitcast<u32>(frame_cbv.data[0].z) ^ bitcast<u32>(frame_cbv.data[0].w);',
      '    let cbv_mod = vec3<f32>(',
      '        f32(b0 & 0xffu) / 255.0,',
      '        f32(b1 & 0xffu) / 255.0,',
      '        f32(b2 & 0xffu) / 255.0,',
      '    );',
      '    let base = vec3<f32>(',
      '        0.08 + 0.35 * uv.x,',
      '        0.14 + 0.40 * uv.y,',
      '        0.24 + 0.30 * (1.0 - uv.x),',
      '    );',
      '    let color = clamp(',
      '        mix(base, cbv_mod, 0.25),',
      '        vec3<f32>(0.0, 0.0, 0.0),',
      '        vec3<f32>(1.0, 1.0, 1.0),',
      '    );',
      '    return vec4<f32>(color, 1.0);',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  /**
   * 24-byte-stride textured quad fallback WGSL pair for splash / UI draws.
   * Vertex attributes: `@location(0) pos: vec4<f32>` (16 B) + `@location(1) uv: vec2<f32>` (8 B).
   * Bindings: `@group(0) @binding(0)` texture_2d<f32> + `@group(0) @binding(1)` sampler.
   */
  const TEXTURED_QUAD_24B_WGSL = Object.freeze({
    vertex: [
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(',
      '    @location(0) pos: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      ') -> VsOut {',
      '    var out: VsOut;',
      '    let w = select(pos.w, 1.0, pos.w == 0.0);',
      '    out.position = vec4<f32>(pos.xyz, w);',
      '    out.uv = uv;',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      '@group(0) @binding(0) var quad_tex: texture_2d<f32>;',
      '@group(0) @binding(1) var quad_sampler: sampler;',
      '',
      '@fragment',
      'fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {',
      '    let texel = textureSample(quad_tex, quad_sampler, uv);',
      '    return texel;',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  /**
   * Real Frame-1 24-byte splash pass fallback WGSL pair for expanded Xbox RECTLIST quads.
   * Vertex attributes (24 B stride):
   *   `@location(0) pos: vec3<f32>` (float32x3 @ offset 0)
   *   `@location(1) color: vec4<f32>` (unorm8x4 @ offset 12)
   *   `@location(2) uv: vec2<f32>` (float32x2 @ offset 16)
   * Bindings:
   *   `@group(0) @binding(0)` uniform FrameCbv (`cb0`)
   *   `@group(0) @binding(1)` texture_2d<f32> (`tex0`)
   *   `@group(0) @binding(2)` sampler (`samp0`)
   */
  const SPLASH_RECT_24B_WGSL = Object.freeze({
    vertex: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> cb0: FrameCbv;',
      '',
      'struct VsOut {',
      '    @builtin(position) position: vec4<f32>,',
      '    @location(0) color: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      '};',
      '',
      '@vertex',
      'fn vs_main(',
      '    @location(0) pos: vec3<f32>,',
      '    @location(1) color: vec4<f32>,',
      '    @location(2) uv: vec2<f32>,',
      ') -> VsOut {',
      '    let cbv_keep = f32(bitcast<u32>(cb0.data[0].x) & 0xffu) * 0.0;',
      '    var out: VsOut;',
      '    out.position = vec4<f32>(pos.xy, pos.z + cbv_keep, 1.0);',
      '    out.color = color;',
      '    out.uv = uv;',
      '    return out;',
      '}',
      '',
    ].join('\n'),
    fragment: [
      'struct FrameCbv {',
      '    data: array<vec4<f32>, 16>,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> cb0: FrameCbv;',
      '@group(0) @binding(1) var tex0: texture_2d<f32>;',
      '@group(0) @binding(2) var samp0: sampler;',
      '',
      'struct FsIn {',
      '    @location(0) color: vec4<f32>,',
      '    @location(1) uv: vec2<f32>,',
      '};',
      '',
      '@fragment',
      'fn fs_main(input: FsIn) -> @location(0) vec4<f32> {',
      '    let texel = textureSample(tex0, samp0, input.uv);',
      '    let cbv_keep = f32(bitcast<u32>(cb0.data[0].x) & 0xffu) * 0.0;',
      '    let tinted = texel * input.color;',
      '    return clamp(',
      '        vec4<f32>(tinted.rgb + vec3<f32>(cbv_keep, cbv_keep, cbv_keep), tinted.a),',
      '        vec4<f32>(0.0, 0.0, 0.0, 0.0),',
      '        vec4<f32>(1.0, 1.0, 1.0, 1.0),',
      '    );',
      '}',
      '',
    ].join('\n'),
    compute: BUILTIN_FALLBACK_WGSL.compute,
  });

  const BUILTIN_FALLBACK_PROFILES = Object.freeze({
    default: BUILTIN_FALLBACK_WGSL,
    fullscreen_triangle_cbv: FULLSCREEN_TRIANGLE_CBV_WGSL,
    textured_quad_24b: TEXTURED_QUAD_24B_WGSL,
    splash_rect_24b: SPLASH_RECT_24B_WGSL,
  });

  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: 'default',
    minimal: 'default',
    fullscreen_triangle_cbv: 'fullscreen_triangle_cbv',
    fullscreen_cbv: 'fullscreen_triangle_cbv',
    'fullscreen-triangle-cbv': 'fullscreen_triangle_cbv',
    'fullscreen-cbv': 'fullscreen_triangle_cbv',
    textured_quad_24b: 'textured_quad_24b',
    textured_24b: 'textured_quad_24b',
    'textured-quad-24b': 'textured_quad_24b',
    'textured-24b': 'textured_quad_24b',
    splash_rect_24b: 'splash_rect_24b',
    'splash-rect-24b': 'splash_rect_24b',
    rectlist_24b: 'splash_rect_24b',
    'rectlist-24b': 'splash_rect_24b',
    frame1_splash_24b: 'splash_rect_24b',
    'frame1-splash-24b': 'splash_rect_24b',
  });

  /**
   * Normalize a built-in fallback profile name or alias to a canonical profile key.
   * Unknown or omitted values safely resolve to `'default'`.
   *
   * @param {string | null | undefined} profile
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
   */
  function normalizeFallbackProfile(profile) {
    if (typeof profile !== 'string') {
      return 'default';
    }
    if (Object.prototype.hasOwnProperty.call(BUILTIN_FALLBACK_PROFILES, profile)) {
      return profile;
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return 'default';
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return 'default';
  }

  /**
   * Build a minimal fullscreen-triangle fallback vertex WGSL shader whose `struct VsOut`
   * matches arbitrary PS `@location(N)` / `@interpolate(...)` / type expectations (defaulting
   * to `@location(0..7)` `vec4<f32>` zero-filled outputs when no explicit locations or `psWgsl`
   * are provided). Ensures fallback VS paired with real PS shaders (including `@interpolate(flat)`
   * and integer locations) passes WebGPU inter-stage validation without errors.
   *
   * @param {{
   *   locations?: Array<{ location: number, type?: string, interpolate?: string | null }>,
   *   psWgsl?: string,
   *   locationCount?: number,
   *   entryPoint?: string,
   * }} [options]
   * @returns {string}
   */
  function buildInterstageFallbackVertexWgsl(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const entryPoint =
      typeof opts.entryPoint === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(opts.entryPoint)
        ? opts.entryPoint
        : 'vs_main';

    const byLocation = new Map();

    const normalizeScalarZero = (typeStr) => {
      const t = String(typeStr || 'vec4<f32>').trim();
      if (t === 'u32') return '0u';
      if (t === 'i32') return '0i';
      if (t === 'f16') return '0.0h';
      if (t === 'f32') return '0.0';
      const vm = /^vec([234])\s*<\s*(f32|f16|i32|u32)\s*>$/.exec(t);
      if (vm) {
        const sc = vm[2];
        const z = sc === 'u32' ? '0u' : sc === 'i32' ? '0i' : sc === 'f16' ? '0.0h' : '0.0';
        return `vec${vm[1]}<${sc}>(${z})`;
      }
      return 'vec4<f32>(0.0)';
    };

    const normalizeIoType = (rawType) => {
      if (typeof rawType !== 'string') return 'vec4<f32>';
      const t = rawType.trim();
      const shorthand = {
        vec2f: 'vec2<f32>',
        vec3f: 'vec3<f32>',
        vec4f: 'vec4<f32>',
        vec2h: 'vec2<f16>',
        vec3h: 'vec3<f16>',
        vec4h: 'vec4<f16>',
        vec2i: 'vec2<i32>',
        vec3i: 'vec3<i32>',
        vec4i: 'vec4<i32>',
        vec2u: 'vec2<u32>',
        vec3u: 'vec3<u32>',
        vec4u: 'vec4<u32>',
      };
      if (Object.prototype.hasOwnProperty.call(shorthand, t)) {
        return shorthand[t];
      }
      if (/^(?:f32|f16|i32|u32)$/.test(t)) {
        return t;
      }
      const vm = /^vec([234])\s*<\s*(f32|f16|i32|u32)\s*>$/.exec(t);
      if (vm) {
        return `vec${vm[1]}<${vm[2]}>`;
      }
      return 'vec4<f32>';
    };

    const isIntegerType = (canonicalType) =>
      canonicalType === 'u32' ||
      canonicalType === 'i32' ||
      canonicalType.includes('<u32>') ||
      canonicalType.includes('<i32>');

    const normalizeInterpAttr = (rawInterp, canonicalType) => {
      if (typeof rawInterp === 'string' && rawInterp.trim().length > 0) {
        const trimmed = rawInterp.trim();
        const inner = trimmed.startsWith('@interpolate')
          ? trimmed.replace(/^@interpolate\s*\(\s*/, '').replace(/\s*\)$/, '')
          : trimmed;
        const parts = inner
          .split(',')
          .map((p) => p.trim().toLowerCase())
          .filter((p) => p.length > 0);
        if (parts.length > 0) {
          const mode = parts[0];
          const sampling = parts.length > 1 ? parts[1] : null;
          if (mode === 'perspective' && (!sampling || sampling === 'center') && !isIntegerType(canonicalType)) {
            return null;
          }
          if (mode === 'linear' && (!sampling || sampling === 'center')) {
            return '@interpolate(linear)';
          }
          return sampling ? `@interpolate(${mode}, ${sampling})` : `@interpolate(${mode})`;
        }
      }
      if (isIntegerType(canonicalType)) {
        return '@interpolate(flat)';
      }
      return null;
    };

    if (typeof opts.psWgsl === 'string' && opts.psWgsl.length > 0) {
      // Use PlayGta6PipelineCache.parseInterstageInterface when loaded, else lightweight regex scan.
      const cacheApi =
        typeof globalThis !== 'undefined' && globalThis.PlayGta6PipelineCache
          ? globalThis.PlayGta6PipelineCache
          : null;
      if (cacheApi && typeof cacheApi.parseInterstageInterface === 'function') {
        const psIface = cacheApi.parseInterstageInterface(opts.psWgsl, 'fragment');
        if (psIface && psIface.locations) {
          for (const [loc, info] of psIface.locations.entries()) {
            if (!Number.isInteger(loc) || loc < 0 || loc > 15) continue;
            const cType = normalizeIoType(info.typeInfo ? info.typeInfo.canonical : info.rawType);
            const iAttr = normalizeInterpAttr(
              info.interpolate && info.interpolate.wgslAttr ? info.interpolate.wgslAttr : null,
              cType
            );
            byLocation.set(loc, { location: loc, type: cType, interpolateAttr: iAttr });
          }
        }
      } else {
        // Lightweight standalone scanner (when PlayGta6PipelineCache is not loaded):
        // strip comments and any fragment output struct named by `@fragment fn ... -> <OutStruct>`.
        const noComments = opts.psWgsl
          .replace(/\/\*[\s\S]*?\*\//g, ' ')
          .replace(/\/\/[^\r\n]*/g, ' ');
        let psScanSrc = noComments;
        const fragRetStructMatch =
          /@fragment\b[\s\S]*?\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{/.exec(
            noComments
          );
        if (fragRetStructMatch) {
          const outStructRe = new RegExp(`\\bstruct\\s+${fragRetStructMatch[1]}\\s*\\{[^}]*\\}`, 'g');
          psScanSrc = psScanSrc.replace(outStructRe, '');
        }
        const declRe =
          /(?:@interpolate\s*\(([^)]+)\)\s*)?@location\s*\(\s*(\d+)u?\s*\)(?:\s*@interpolate\s*\(([^)]+)\))?\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*(vec[234]\s*<\s*(?:f32|f16|i32|u32)\s*>|vec[234][fhiu]|f32|f16|i32|u32)/g;
        let m;
        while ((m = declRe.exec(psScanSrc)) !== null) {
          const loc = Number(m[2]);
          if (!Number.isInteger(loc) || loc < 0 || loc > 15) continue;
          const cType = normalizeIoType(m[4]);
          const iAttr = normalizeInterpAttr(m[1] || m[3] || null, cType);
          if (!byLocation.has(loc)) {
            byLocation.set(loc, { location: loc, type: cType, interpolateAttr: iAttr });
          }
        }
      }
    }

    if (Array.isArray(opts.locations)) {
      for (let i = 0; i < opts.locations.length; i += 1) {
        const spec = opts.locations[i];
        if (!spec || typeof spec !== 'object') continue;
        const loc = spec.location;
        if (!Number.isInteger(loc) || loc < 0 || loc > 15) continue;
        const cType = normalizeIoType(spec.type);
        const iAttr = normalizeInterpAttr(spec.interpolate, cType);
        byLocation.set(loc, { location: loc, type: cType, interpolateAttr: iAttr });
      }
    }

    const hasExplicitInputs =
      typeof opts.psWgsl === 'string' && opts.psWgsl.length > 0
        ? true
        : Array.isArray(opts.locations);

    if (!hasExplicitInputs) {
      const count =
        Number.isInteger(opts.locationCount) && opts.locationCount >= 0
          ? Math.min(16, opts.locationCount)
          : 8;
      for (let loc = 0; loc < count; loc += 1) {
        byLocation.set(loc, { location: loc, type: 'vec4<f32>', interpolateAttr: null });
      }
    }

    const sortedLocs = Array.from(byLocation.values()).sort((a, b) => a.location - b.location);
    const structFields = ['    @builtin(position) position: vec4<f32>,'];
    const assignLines = [];
    for (let i = 0; i < sortedLocs.length; i += 1) {
      const entry = sortedLocs[i];
      const fieldName = `param_${entry.location}_`;
      const attrPart = entry.interpolateAttr
        ? `@location(${entry.location}) ${entry.interpolateAttr}`
        : `@location(${entry.location})`;
      structFields.push(`    ${attrPart} ${fieldName}: ${entry.type},`);
      assignLines.push(`    out.${fieldName} = ${normalizeScalarZero(entry.type)};`);
    }

    return [
      'struct VsOut {',
      ...structFields,
      '};',
      '',
      '@vertex',
      `fn ${entryPoint}(@builtin(vertex_index) vertex_index: u32) -> VsOut {`,
      '    var positions = array<vec2<f32>, 3>(',
      '        vec2<f32>(-1.0, -1.0),',
      '        vec2<f32>(3.0, -1.0),',
      '        vec2<f32>(-1.0, 3.0),',
      '    );',
      '    let idx = vertex_index % 3u;',
      '    var out: VsOut;',
      '    out.position = vec4<f32>(positions[idx], 0.0, 1.0);',
      ...assignLines,
      '    return out;',
      '}',
      '',
    ].join('\n');
  }

  /**
   * Return a sanitized, JSON-serializable map of available built-in fallback WGSL profiles.
   * Each call returns fresh copies so caller mutations cannot alter router templates.
   *
   * @returns {Record<string, { vertex: string, fragment: string, compute: string }>}
   */
  function getBuiltInFallbackProfiles() {
    const out = {};
    const names = Object.keys(BUILTIN_FALLBACK_PROFILES);
    for (let i = 0; i < names.length; i += 1) {
      const name = names[i];
      const entry = BUILTIN_FALLBACK_PROFILES[name];
      out[name] = {
        vertex: entry.vertex,
        fragment: entry.fragment,
        compute: entry.compute,
      };
    }
    return out;
  }

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  /**
   * Normalize a 64-bit shader hash (`BigInt`, `0x...` hex, bare hex, or safe non-negative
   * integer) into an unsigned 64-bit `BigInt` matching `normalizeHash` / `normalizeShaderHash`.
   *
   * @param {bigint | string | number} hash
   * @returns {bigint}
   */
  function normalizeShaderHash(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }

    if (typeof hash === 'number') {
      if (!Number.isInteger(hash) || !Number.isSafeInteger(hash) || hash < 0) {
        throw new RangeError(`Numeric shader hash must be a non-negative safe integer, got: ${hash}`);
      }
      return BigInt(hash);
    }

    if (typeof hash === 'string') {
      const trimmed = hash.trim();
      const rawHex =
        trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
      if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
        throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
      }
      return BigInt(`0x${rawHex}`);
    }

    throw new TypeError('Shader hash must be a BigInt, hex string, or non-negative safe integer');
  }

  /**
   * Format a 64-bit shader hash as canonical `0x` + 16 lowercase hex digits.
   *
   * @param {bigint | string | number} hash
   * @returns {string}
   */
  function formatShaderHash(hash) {
    const value = normalizeShaderHash(hash);
    return `0x${value.toString(16).padStart(16, '0')}`;
  }

  /**
   * Normalize a shader stage code or name to canonical `"vertex" | "fragment" | "compute"`.
   *
   * @param {number | string} stage
   * @returns {'vertex' | 'fragment' | 'compute'}
   */
  function normalizeStage(stage) {
    if (typeof stage === 'number') {
      const name = STAGE_NAMES[stage];
      if (name !== undefined) {
        return name;
      }
      throw new RangeError(`Invalid shader stage code: ${stage}`);
    }
    if (typeof stage === 'string') {
      const lower = stage.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(STAGE_NAME_TO_CODE, lower)) {
        return STAGE_NAMES[STAGE_NAME_TO_CODE[lower]];
      }
      throw new RangeError(`Invalid shader stage name: ${JSON.stringify(stage)}`);
    }
    throw new TypeError('Shader stage must be a stage code (1..3) or stage name');
  }

  /**
   * Zero-allocation hot-path stage code resolver (Vertex=1, Fragment=2, Compute=3).
   * Falls back to `normalizeStage` for trimmed/mixed-case strings and validation errors.
   *
   * @param {number | string} stage
   * @returns {1 | 2 | 3}
   */
  function normalizeStageCode(stage) {
    if (stage === SHADER_STAGE.VERTEX || stage === 'vertex' || stage === 'vs') {
      return SHADER_STAGE.VERTEX;
    }
    if (
      stage === SHADER_STAGE.FRAGMENT ||
      stage === 'fragment' ||
      stage === 'fs' ||
      stage === 'ps' ||
      stage === 'pixel'
    ) {
      return SHADER_STAGE.FRAGMENT;
    }
    if (stage === SHADER_STAGE.COMPUTE || stage === 'compute' || stage === 'cs') {
      return SHADER_STAGE.COMPUTE;
    }
    return STAGE_NAME_TO_CODE[normalizeStage(stage)];
  }

  function normalizeFrameNumber(frame) {
    if (
      typeof frame !== 'number' ||
      !Number.isInteger(frame) ||
      !Number.isSafeInteger(frame) ||
      frame < 0
    ) {
      throw new RangeError(`frame must be a non-negative safe integer, got ${frame}`);
    }
    return frame;
  }

  /**
   * Normalize a missing-shader reason string to one of:
   * `'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other'`.
   *
   * @param {string | null | undefined} reason
   * @returns {'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other'}
   */
  function normalizeMissingReason(reason) {
    if (typeof reason !== 'string') {
      return MISSING_SHADER_REASONS.OTHER;
    }
    const key = reason
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, '_');
    if (key.length === 0) {
      return MISSING_SHADER_REASONS.OTHER;
    }
    if (Object.prototype.hasOwnProperty.call(MISSING_SHADER_REASON_ALIASES, key)) {
      return MISSING_SHADER_REASON_ALIASES[key];
    }
    return MISSING_SHADER_REASONS.OTHER;
  }

  /**
   * Sanitize a negative compile-failure reason into a short, safe identifier token
   * (e.g. `'compile_failed'`, `'missing_pack_entry'`, `'tint_validation_failed'`).
   * Never preserves host paths, raw WGSL snippets, or unbounded error strings.
   *
   * @param {string | null | undefined} reason
   * @returns {string}
   */
  function sanitizeCompileFailureReason(reason) {
    if (typeof reason !== 'string') {
      return MISSING_SHADER_REASONS.COMPILE_FAILED;
    }
    const normalized = reason
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, '_');
    if (normalized.length === 0) {
      return MISSING_SHADER_REASONS.COMPILE_FAILED;
    }
    if (SAFE_REASON_TOKEN_REGEX.test(normalized)) {
      return normalized;
    }
    const mapped = normalizeMissingReason(reason);
    return mapped === MISSING_SHADER_REASONS.OTHER
      ? MISSING_SHADER_REASONS.COMPILE_FAILED
      : mapped;
  }

  /**
   * Deterministically select a built-in fallback WGSL profile from shader stage and
   * binding signature:
   * - compute stage -> `'default'` (compute no-op)
   * - explicit `profile` hint -> `normalizeFallbackProfile(profile)`
   * - vertex-colored UI / CBV+texture splash pass -> `'splash_rect_24b'`
   * - textured UI quad / 24-byte vertex stride -> `'textured_quad_24b'`
   * - fullscreen pass with CBV -> `'fullscreen_triangle_cbv'`
   * - otherwise -> `'default'`
   *
   * @param {number | string} stage
   * @param {object | null | undefined} [bindingSignature]
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
   */
  function selectDeterministicFallbackProfile(stage, bindingSignature) {
    const stageCode = normalizeStageCode(stage);
    if (stageCode === SHADER_STAGE.COMPUTE) {
      return 'default';
    }
    if (!bindingSignature || typeof bindingSignature !== 'object') {
      return 'default';
    }
    if (
      typeof bindingSignature.profile === 'string' &&
      bindingSignature.profile.trim().length > 0
    ) {
      return normalizeFallbackProfile(bindingSignature.profile);
    }

    const hasCbv =
      bindingSignature.hasCbv === true ||
      bindingSignature.hasBoundCbv === true ||
      bindingSignature.hasUniform === true ||
      (typeof bindingSignature.cbvCount === 'number' && bindingSignature.cbvCount > 0) ||
      (typeof bindingSignature.uniformCount === 'number' && bindingSignature.uniformCount > 0);

    const hasTexture =
      bindingSignature.hasTexture === true ||
      bindingSignature.hasBoundTexture === true ||
      bindingSignature.hasSampledTexture === true ||
      bindingSignature.hasSrv === true ||
      bindingSignature.hasBoundSrv === true ||
      (typeof bindingSignature.textureCount === 'number' && bindingSignature.textureCount > 0) ||
      (typeof bindingSignature.srvCount === 'number' && bindingSignature.srvCount > 0);

    const hasSampler =
      bindingSignature.hasSampler === true ||
      bindingSignature.hasBoundSampler === true ||
      (typeof bindingSignature.samplerCount === 'number' && bindingSignature.samplerCount > 0);

    const hasVertexColor =
      bindingSignature.hasVertexColor === true ||
      bindingSignature.hasColor === true ||
      bindingSignature.hasColorAttribute === true ||
      bindingSignature.vertexColored === true ||
      (typeof bindingSignature.attributeCount === 'number' && bindingSignature.attributeCount >= 3);

    const hasVertexStride24 =
      bindingSignature.vertexStride === 24 ||
      bindingSignature.arrayStride === 24 ||
      bindingSignature.stride === 24;

    const isFullscreen =
      bindingSignature.fullscreen === true ||
      bindingSignature.isFullscreen === true ||
      bindingSignature.vertexCount === 3;

    if (hasVertexColor || (hasCbv && (hasTexture || hasSampler))) {
      return 'splash_rect_24b';
    }
    if (hasTexture || hasSampler || hasVertexStride24) {
      return 'textured_quad_24b';
    }
    if (hasCbv || isFullscreen) {
      return 'fullscreen_triangle_cbv';
    }
    return 'default';
  }

  function compareLex(a, b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  function createEmptyReasonCounts() {
    return {
      not_in_pack: 0,
      reflection_failed: 0,
      compile_failed: 0,
      other: 0,
    };
  }

  function createEmptyStageCounts() {
    return {
      vertex: 0,
      fragment: 0,
      compute: 0,
    };
  }

  function getDominantMissingReason(row) {
    let bestReason = row.lastReason || MISSING_SHADER_REASONS.OTHER;
    let bestCount = row.reasons[bestReason] || 0;
    for (let i = 0; i < MISSING_SHADER_REASON_ORDER.length; i += 1) {
      const candidate = MISSING_SHADER_REASON_ORDER[i];
      const count = row.reasons[candidate] || 0;
      if (count > bestCount) {
        bestCount = count;
        bestReason = candidate;
      }
    }
    return bestReason;
  }

  function extractEntryPoint(wgsl, stageName) {
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      return DEFAULT_ENTRY_POINTS[stageName] || null;
    }
    const re = ENTRY_POINT_REGEXES[stageName];
    if (re) {
      const match = re.exec(wgsl);
      if (match) {
        return match[1];
      }
    }
    const fallback = FALLBACK_FN_REGEX.exec(wgsl);
    return fallback ? fallback[1] : DEFAULT_ENTRY_POINTS[stageName] || null;
  }

  function isBetterMissingRow(a, b) {
    if (a.hitCount !== b.hitCount) {
      return a.hitCount > b.hitCount;
    }
    if (a.hash !== b.hash) {
      return a.hash < b.hash;
    }
    return a.stage < b.stage;
  }

  function compareMissingRowsDesc(a, b) {
    if (a.hitCount !== b.hitCount) {
      return b.hitCount - a.hitCount;
    }
    if (a.hash !== b.hash) {
      return a.hash < b.hash ? -1 : 1;
    }
    return a.stage < b.stage ? -1 : a.stage > b.stage ? 1 : 0;
  }

  function siftUpMinHeap(heap, index) {
    let i = index;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      // Min-heap puts the WORST (lowest-priority) row at heap[0].
      if (isBetterMissingRow(heap[parent], heap[i])) {
        const tmp = heap[parent];
        heap[parent] = heap[i];
        heap[i] = tmp;
        i = parent;
      } else {
        break;
      }
    }
  }

  function siftDownMinHeap(heap, index) {
    const len = heap.length;
    let i = index;
    while (true) {
      const left = (i << 1) + 1;
      const right = left + 1;
      let worst = i;
      if (left < len && isBetterMissingRow(heap[worst], heap[left])) {
        worst = left;
      }
      if (right < len && isBetterMissingRow(heap[worst], heap[right])) {
        worst = right;
      }
      if (worst === i) {
        break;
      }
      const tmp = heap[i];
      heap[i] = heap[worst];
      heap[worst] = tmp;
      i = worst;
    }
  }

  /**
   * Select the top-K missing-shader rows across stage maps using a bounded min-heap
   * when `limit < totalCount`, avoiding full-table sorting and excess allocations.
   *
   * @param {Array<Map<bigint, object> | null>} missingByStage
   * @param {number} totalCount
   * @param {number} limit
   * @returns {Array<object>}
   */
  function selectTopMissingRows(missingByStage, totalCount, limit) {
    if (limit <= 0 || totalCount <= 0) {
      return [];
    }
    const k = limit < totalCount ? limit : totalCount;
    if (k >= totalCount) {
      const all = new Array(totalCount);
      let idx = 0;
      for (let s = 1; s <= 3; s += 1) {
        const map = missingByStage[s];
        for (const row of map.values()) {
          all[idx] = row;
          idx += 1;
        }
      }
      all.sort(compareMissingRowsDesc);
      return all;
    }

    const heap = [];
    for (let s = 1; s <= 3; s += 1) {
      const map = missingByStage[s];
      for (const row of map.values()) {
        if (heap.length < k) {
          heap.push(row);
          siftUpMinHeap(heap, heap.length - 1);
        } else if (isBetterMissingRow(row, heap[0])) {
          heap[0] = row;
          siftDownMinHeap(heap, 0);
        }
      }
    }
    heap.sort(compareMissingRowsDesc);
    return heap;
  }

  function tryCallPackMethod(packLoader, fn, key) {
    if (typeof fn !== 'function') {
      return undefined;
    }
    try {
      return unwrapPackResult(fn.call(packLoader, key));
    } catch (_err) {
      return undefined;
    }
  }

  function lookupPackSourceWithKey(packLoader, key) {
    return (
      tryCallPackMethod(packLoader, packLoader.getWgsl, key) ||
      tryCallPackMethod(packLoader, packLoader.getSource, key) ||
      tryCallPackMethod(packLoader, packLoader.getShaderSource, key) ||
      tryCallPackMethod(packLoader, packLoader.lookup, key) ||
      tryCallPackMethod(packLoader, packLoader.get, key)
    );
  }

  function unwrapPackResult(raw) {
    if (raw === undefined || raw === null) {
      return undefined;
    }
    if (typeof raw === 'string') {
      return raw.trim().length > 0 ? { wgsl: raw, entryPoint: null } : undefined;
    }
    if (typeof raw === 'object') {
      if (typeof raw.wgsl === 'string' && raw.wgsl.trim().length > 0) {
        return {
          wgsl: raw.wgsl,
          entryPoint:
            typeof raw.entryPoint === 'string'
              ? raw.entryPoint
              : typeof raw.entry_point === 'string'
                ? raw.entry_point
                : null,
        };
      }
      if (typeof raw.source === 'string' && raw.source.trim().length > 0) {
        return { wgsl: raw.source, entryPoint: null };
      }
      if (typeof raw.code === 'string' && raw.code.trim().length > 0) {
        return { wgsl: raw.code, entryPoint: null };
      }
    }
    return undefined;
  }

  /**
   * Look up WGSL from a pack loader supporting getWgsl / getSource / lookup / Map.get
   * (plus getShaderSource for WgslPackCache compatibility).
   *
   * @param {object | Map | null | undefined} packLoader
   * @param {bigint} hashBigInt
   * @returns {{ wgsl: string, entryPoint: string | null } | undefined}
   */
  function lookupPackSource(packLoader, hashBigInt, hashHex) {
    if (!packLoader) {
      return undefined;
    }

    const byBigInt = lookupPackSourceWithKey(packLoader, hashBigInt);
    if (byBigInt) {
      return byBigInt;
    }

    const resolvedHex = hashHex !== undefined ? hashHex : formatShaderHash(hashBigInt);
    return lookupPackSourceWithKey(packLoader, resolvedHex);
  }

  /**
   * Return true when `hashBigInt` exists in a ranged-mode WGPK index whose payload
   * has not yet been fetched into memory. Transient misses for ranged entries must not
   * be negatively memoized so the next `resolve()` immediately picks up the real pack
   * shader once the HTTP 206 Range fetch completes.
   *
   * @param {object | Map | null | undefined} packLoader
   * @param {bigint} hashBigInt
   * @returns {boolean}
   */
  function isPendingRangedPackShader(packLoader, hashBigInt, hashHex) {
    if (!packLoader || typeof packLoader !== 'object') {
      return false;
    }
    if (packLoader._shaderEntries instanceof Map) {
      const record = packLoader._shaderEntries.get(hashBigInt);
      return Boolean(record && record.pack && record.pack.mode === 'ranged');
    }
    if (packLoader.mode === 'ranged' && packLoader.entries instanceof Map) {
      return packLoader.entries.has(hashBigInt);
    }
    if (typeof packLoader.getWgslAsync === 'function') {
      let isRangedMode = packLoader.mode === 'ranged' || packLoader.mode === 'mixed';
      if (!isRangedMode && typeof packLoader.getStats === 'function') {
        try {
          const st = packLoader.getStats();
          const stMode = st && typeof st.mode === 'string' ? st.mode.trim().toLowerCase() : '';
          isRangedMode = stMode === 'ranged' || stMode === 'mixed';
        } catch (_err) {
          isRangedMode = false;
        }
      }
      if (
        !isRangedMode &&
        typeof packLoader.getWgsl !== 'function' &&
        typeof packLoader.getSource !== 'function' &&
        typeof packLoader.getShaderSource !== 'function' &&
        typeof packLoader.lookup !== 'function' &&
        typeof packLoader.get !== 'function'
      ) {
        isRangedMode = true;
      }
      if (isRangedMode) {
        const hasFn =
          typeof packLoader.hasShader === 'function'
            ? packLoader.hasShader
            : typeof packLoader.has === 'function'
              ? packLoader.has
              : null;
        if (hasFn) {
          try {
            if (hasFn.call(packLoader, hashBigInt)) {
              return true;
            }
          } catch (_err) {
            // Fall through to canonical hex lookup.
          }
          const resolvedHex = hashHex !== undefined ? hashHex : formatShaderHash(hashBigInt);
          try {
            if (hasFn.call(packLoader, resolvedHex)) {
              return true;
            }
          } catch (_err) {
            return false;
          }
        }
      }
    }
    return false;
  }

  function resolveFallbackHashForStage(fallbackHashes, stageName) {
    if (!fallbackHashes || typeof fallbackHashes !== 'object') {
      return null;
    }
    const raw =
      fallbackHashes[stageName] !== undefined
        ? fallbackHashes[stageName]
        : fallbackHashes[STAGE_NAME_TO_CODE[stageName]];
    if (raw === undefined || raw === null || raw === '') {
      return null;
    }
    return normalizeShaderHash(raw);
  }

  /**
   * Missing-shader fallback router used by the WebGPU worker during early M3/M4 frames.
   */
  class ShaderFallbackRouter {
    /**
     * @param {{
     *   packLoader?: object | Map | null,
     *   fallbackHashes?: Record<string, bigint | string | number> | null,
     *   maxTrackedMissingHashes?: number,
     *   maxTrackedMissing?: number,
     *   maxMissingHashes?: number,
     *   maxCategorizedMissingHashes?: number,
     *   maxResolveCacheEntries?: number,
     *   maxCompileFailureEntries?: number,
     *   initialFrame?: number,
     *   defaultProfile?: string | null,
     *   realShaders?: Iterable<any> | Map<any, any> | Record<string, any> | null,
     *   now?: () => number,
     * }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('ShaderFallbackRouter options must be an object');
      }
      const opts = options || {};

      this._packLoader = opts.packLoader !== undefined ? opts.packLoader : null;
      this._fallbackHashes =
        opts.fallbackHashes && typeof opts.fallbackHashes === 'object'
          ? { ...opts.fallbackHashes }
          : {};

      let maxTracked = DEFAULT_MAX_TRACKED_MISSING_HASHES;
      let hasExplicitLegacyCap = false;
      if (opts.maxTrackedMissingHashes !== undefined) {
        if (
          !Number.isInteger(opts.maxTrackedMissingHashes) ||
          opts.maxTrackedMissingHashes < 0
        ) {
          throw new RangeError(
            `maxTrackedMissingHashes must be a non-negative integer, got ${opts.maxTrackedMissingHashes}`
          );
        }
        maxTracked = opts.maxTrackedMissingHashes;
        hasExplicitLegacyCap = true;
      } else if (opts.maxTrackedMissing !== undefined) {
        if (!Number.isInteger(opts.maxTrackedMissing) || opts.maxTrackedMissing < 0) {
          throw new RangeError(
            `maxTrackedMissing must be a non-negative integer, got ${opts.maxTrackedMissing}`
          );
        }
        maxTracked = opts.maxTrackedMissing;
        hasExplicitLegacyCap = true;
      }
      this._maxTrackedMissing = maxTracked;

      let maxMissingHashes = DEFAULT_MAX_MISSING_HASHES;
      if (opts.maxMissingHashes !== undefined) {
        if (!Number.isInteger(opts.maxMissingHashes) || opts.maxMissingHashes < 0) {
          throw new RangeError(
            `maxMissingHashes must be a non-negative integer, got ${opts.maxMissingHashes}`
          );
        }
        maxMissingHashes = opts.maxMissingHashes;
      }
      this._maxMissingHashes = maxMissingHashes;

      let maxCategorized = hasExplicitLegacyCap
        ? maxTracked
        : DEFAULT_MAX_CATEGORIZED_MISSING_HASHES;
      if (opts.maxCategorizedMissingHashes !== undefined) {
        if (
          !Number.isInteger(opts.maxCategorizedMissingHashes) ||
          opts.maxCategorizedMissingHashes < 0
        ) {
          throw new RangeError(
            `maxCategorizedMissingHashes must be a non-negative integer, got ${opts.maxCategorizedMissingHashes}`
          );
        }
        maxCategorized = opts.maxCategorizedMissingHashes;
      }
      this._maxCategorizedMissing = maxCategorized;

      let maxResolveCache = DEFAULT_MAX_RESOLVE_CACHE_ENTRIES;
      if (opts.maxResolveCacheEntries !== undefined) {
        if (
          !Number.isInteger(opts.maxResolveCacheEntries) ||
          opts.maxResolveCacheEntries < 0
        ) {
          throw new RangeError(
            `maxResolveCacheEntries must be a non-negative integer, got ${opts.maxResolveCacheEntries}`
          );
        }
        maxResolveCache = opts.maxResolveCacheEntries;
      }
      this._maxResolveCacheEntries = maxResolveCache;

      let maxResolveCachePackWgslBytes = DEFAULT_MAX_RESOLVE_CACHE_PACK_WGSL_BYTES;
      if (opts.maxResolveCachePackWgslBytes !== undefined) {
        if (
          !Number.isInteger(opts.maxResolveCachePackWgslBytes) ||
          opts.maxResolveCachePackWgslBytes < 0
        ) {
          throw new RangeError(
            `maxResolveCachePackWgslBytes must be a non-negative integer, got ${opts.maxResolveCachePackWgslBytes}`
          );
        }
        maxResolveCachePackWgslBytes = opts.maxResolveCachePackWgslBytes;
      }
      this._maxResolveCachePackWgslBytes = maxResolveCachePackWgslBytes;
      this._resolveCachePackWgslBytes = 0;

      let maxRealShaders = DEFAULT_MAX_REAL_SHADERS;
      if (opts.maxRealShaders !== undefined) {
        if (!Number.isInteger(opts.maxRealShaders) || opts.maxRealShaders < 1) {
          throw new RangeError(
            `maxRealShaders must be a positive integer, got ${opts.maxRealShaders}`
          );
        }
        maxRealShaders = opts.maxRealShaders;
      }
      this._maxRealShaders = maxRealShaders;

      let maxCompileFailures = DEFAULT_MAX_COMPILE_FAILURE_ENTRIES;
      if (opts.maxCompileFailureEntries !== undefined) {
        if (
          !Number.isInteger(opts.maxCompileFailureEntries) ||
          opts.maxCompileFailureEntries < 0
        ) {
          throw new RangeError(
            `maxCompileFailureEntries must be a non-negative integer, got ${opts.maxCompileFailureEntries}`
          );
        }
        maxCompileFailures = opts.maxCompileFailureEntries;
      }
      this._maxCompileFailureEntries = maxCompileFailures;

      const initialFrame =
        opts.initialFrame !== undefined && opts.initialFrame !== null
          ? normalizeFrameNumber(opts.initialFrame)
          : 0;
      this._currentFrame = initialFrame;

      this._now = typeof opts.now === 'function' ? opts.now : defaultNow;
      this._defaultProfile = normalizeFallbackProfile(opts.defaultProfile);

      // W204: Per-stage O(1) missing-hash maps keyed by u64 BigInt (zero key-concatenation).
      this._missingByStage = [null, new Map(), new Map(), new Map()];
      this._missingCount = 0;
      this._overflowMissingCount = 0;
      this._telemetryGeneration = 0;

      // W304: Bounded sanitized missing/fallback shader frequency table keyed by u64 BigInt
      // for S47 `playgta6-shaders pack-rdna2 --priority-hashes` export.
      this._missingPriorityHashes = new Map();
      this._missingPrioritySeq = 0;
      this._missingPriorityEvictions = 0;

      // W204/W304: Fast-path per-stage resolved-hash memoization (zero temporary objects / zero hex formatting on cache hits).
      this._resolveCacheByStage = [null, new Map(), new Map(), new Map()];
      this._resolveCacheEntries = new Map();
      this._resolveCacheHits = 0;
      this._fastPathHits = 0;
      this._resolveCacheMisses = 0;
      this._resolveCacheInvalidations = 0;
      this._packStampVersion = undefined;
      this._packStampNextId = -1;
      this._packStampPacksSize = -1;
      this._packStampEntriesSize = -1;
      this._packStampDynamicSize = -1;
      this._packStampMapSize = -1;
      this._capturePackStamp();

      this._stats = {
        totalLookups: 0,
        packHits: 0,
        fallbackHits: 0,
        builtinHits: 0,
      };
      this._rangedPendingHits = 0;

      // W179: Categorized missing-shader telemetry (bounded unique hash+stage entries).
      this._categorizedMissing = new Map();
      this._categorizedOverflowCount = 0;
      this._categorizedTotals = {
        total: 0,
        byReason: createEmptyReasonCounts(),
        byStage: createEmptyStageCounts(),
      };
      // W363: Monotonic mutation version + bounded cache for getMissingShaderSummary().
      this._summaryVersion = 0;
      this._summaryCacheVersion = -1;
      this._summaryCache = new Map();
      this._summaryUniqueCountsCache = null;

      // W179: Real-shader override guard (protects real pack, menu, and Bink2 YUV shaders).
      this._realShaders = new Map();
      this._blockedOverrides = new Map();
      this._blockedOverrideCount = 0;
      this._blockedOverflowCount = 0;

      // W268: Bounded negative compile-failure LRU cache keyed by canonical normalized hash.
      this._compileFailures = new Map();
      this._compileFailureCacheHits = 0;
      this._compileFailureRecorded = 0;

      if (opts.realShaders !== undefined && opts.realShaders !== null) {
        this._registerInitialRealShaders(opts.realShaders);
        this._resolveCacheInvalidations = 0;
      }
    }

    _capturePackStamp() {
      const loader = this._packLoader;
      if (loader === null || loader === undefined) {
        this._packStampVersion = undefined;
        this._packStampNextId = -1;
        this._packStampPacksSize = -1;
        this._packStampEntriesSize = -1;
        this._packStampDynamicSize = -1;
        this._packStampMapSize = -1;
        return;
      }
      this._packStampVersion =
        loader.generation !== undefined
          ? loader.generation
          : loader.packVersion !== undefined
            ? loader.packVersion
            : loader.version;
      this._packStampNextId = typeof loader._nextPackId === 'number' ? loader._nextPackId : -1;
      this._packStampPacksSize = loader._packs instanceof Map ? loader._packs.size : -1;
      this._packStampEntriesSize =
        loader._shaderEntries instanceof Map
          ? loader._shaderEntries.size
          : loader._entries instanceof Map
            ? loader._entries.size
            : -1;
      this._packStampDynamicSize =
        loader._dynamicEntries instanceof Map ? loader._dynamicEntries.size : -1;
      this._packStampMapSize = typeof loader.size === 'number' ? loader.size : -1;
    }

    _syncPackCacheVersion() {
      const loader = this._packLoader;
      if (loader === null || loader === undefined) {
        return;
      }
      const v =
        loader.generation !== undefined
          ? loader.generation
          : loader.packVersion !== undefined
            ? loader.packVersion
            : loader.version;
      const nextId = typeof loader._nextPackId === 'number' ? loader._nextPackId : -1;
      const packsSize = loader._packs instanceof Map ? loader._packs.size : -1;
      const entriesSize =
        loader._shaderEntries instanceof Map
          ? loader._shaderEntries.size
          : loader._entries instanceof Map
            ? loader._entries.size
            : -1;
      const dynSize = loader._dynamicEntries instanceof Map ? loader._dynamicEntries.size : -1;
      const mapSize = typeof loader.size === 'number' ? loader.size : -1;

      if (
        v !== this._packStampVersion ||
        nextId !== this._packStampNextId ||
        packsSize !== this._packStampPacksSize ||
        entriesSize !== this._packStampEntriesSize ||
        dynSize !== this._packStampDynamicSize ||
        mapSize !== this._packStampMapSize
      ) {
        this._packStampVersion = v;
        this._packStampNextId = nextId;
        this._packStampPacksSize = packsSize;
        this._packStampEntriesSize = entriesSize;
        this._packStampDynamicSize = dynSize;
        this._packStampMapSize = mapSize;
        this._clearResolveCache();
      }
    }

    _clearResolveCache() {
      this._resolveCacheByStage[1].clear();
      this._resolveCacheByStage[2].clear();
      this._resolveCacheByStage[3].clear();
      this._resolveCacheEntries.clear();
      this._resolveCachePackWgslBytes = 0;
      this._resolveCacheInvalidations += 1;
    }

    _evictResolveCacheEntry(entry) {
      const stageCache = this._resolveCacheByStage[entry.stageCode];
      if (entry.key1 !== undefined && stageCache.get(entry.key1) === entry) {
        stageCache.delete(entry.key1);
      }
      if (entry.key2 !== undefined && stageCache.get(entry.key2) === entry) {
        stageCache.delete(entry.key2);
      }
      if (entry.key3 !== undefined && stageCache.get(entry.key3) === entry) {
        stageCache.delete(entry.key3);
      }
      if (entry.key4 !== undefined && stageCache.get(entry.key4) === entry) {
        stageCache.delete(entry.key4);
      }
      if (this._resolveCacheEntries.delete(entry) && entry.packWgslBytes > 0) {
        this._resolveCachePackWgslBytes = Math.max(
          0,
          this._resolveCachePackWgslBytes - entry.packWgslBytes
        );
      }
    }

    _touchResolveCacheEntry(entry) {
      if (this._resolveCacheEntries.delete(entry)) {
        this._resolveCacheEntries.set(entry, entry);
      }
    }

    _evictResolveCacheForHash(hashBigInt, hashHex) {
      for (let s = 1; s <= 3; s += 1) {
        const stageCache = this._resolveCacheByStage[s];
        const entry = stageCache.get(hashBigInt) || stageCache.get(hashHex);
        if (entry !== undefined) {
          this._evictResolveCacheEntry(entry);
        }
      }
    }

    _recordCompileFailureEntry(hashBigInt, hashHex, stageName, reason, nowMs) {
      const sanitizedReason = sanitizeCompileFailureReason(reason);
      this._compileFailureRecorded += 1;
      if (this._isConfiguredFallbackHash(hashBigInt)) {
        this._clearResolveCache();
      } else {
        this._evictResolveCacheForHash(hashBigInt, hashHex);
      }

      if (this._maxCompileFailureEntries <= 0) {
        return null;
      }

      const existing = this._compileFailures.get(hashBigInt);
      if (existing !== undefined) {
        existing.count += 1;
        existing.reason = sanitizedReason;
        if (stageName !== null) {
          existing.stage = stageName;
        }
        existing.lastSeenMs = nowMs;
        // Refresh LRU position (move to tail of Map).
        this._compileFailures.delete(hashBigInt);
        this._compileFailures.set(hashBigInt, existing);
        return existing;
      }

      if (this._compileFailures.size >= this._maxCompileFailureEntries) {
        const oldestKey = this._compileFailures.keys().next().value;
        if (oldestKey !== undefined) {
          this._compileFailures.delete(oldestKey);
        }
      }

      const entry = {
        hashBigInt,
        hashHex,
        stage: stageName,
        reason: sanitizedReason,
        count: 1,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      };
      this._compileFailures.set(hashBigInt, entry);
      return entry;
    }

    /**
     * Record a shader hash into the bounded negative compile-failure LRU cache so
     * subsequent `hasCompileFailure()` / `resolve()` calls skip retrying broken WGSL
     * and route straight to a deterministic fallback profile.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | { stage?: number | string | null, reason?: string | null } | null} [stageOrOpts]
     * @param {string | null} [maybeReason]
     * @returns {{
     *   hashHex: string,
     *   stage: 'vertex' | 'fragment' | 'compute' | null,
     *   reason: string,
     *   count: number,
     *   firstSeenMs: number,
     *   lastSeenMs: number,
     *   cached: boolean,
     * }}
     */
    recordCompileFailure(hash, stageOrOpts = null, maybeReason) {
      const hashBigInt = normalizeShaderHash(hash);
      const hashHex = formatShaderHash(hashBigInt);
      let stageArg = stageOrOpts;
      let reasonArg = maybeReason;

      if (
        maybeReason === undefined &&
        stageOrOpts !== null &&
        typeof stageOrOpts === 'object'
      ) {
        stageArg = stageOrOpts.stage !== undefined ? stageOrOpts.stage : null;
        reasonArg = stageOrOpts.reason;
      }

      const stageName =
        stageArg === undefined || stageArg === null ? null : normalizeStage(stageArg);
      const sanitizedReason = sanitizeCompileFailureReason(reasonArg);
      const nowMs = this._now();
      const entry = this._recordCompileFailureEntry(
        hashBigInt,
        hashHex,
        stageName,
        sanitizedReason,
        nowMs
      );
      if (entry === null) {
        return {
          hashHex,
          stage: stageName,
          reason: sanitizedReason,
          count: 1,
          firstSeenMs: nowMs,
          lastSeenMs: nowMs,
          cached: false,
        };
      }
      return {
        hashHex: entry.hashHex,
        stage: entry.stage,
        reason: entry.reason,
        count: entry.count,
        firstSeenMs: entry.firstSeenMs,
        lastSeenMs: entry.lastSeenMs,
        cached: true,
      };
    }

    /**
     * Return true when `hash` is present in the negative compile-failure cache.
     * Touches the entry for LRU recency and increments `compileFailureCacheHits` on hits.
     *
     * @param {bigint | string | number | null | undefined} hash
     * @returns {boolean}
     */
    hasCompileFailure(hash) {
      if (hash === null || hash === undefined || hash === '') {
        return false;
      }
      const hashBigInt = normalizeShaderHash(hash);
      const entry = this._compileFailures.get(hashBigInt);
      if (entry === undefined) {
        return false;
      }
      // Refresh LRU recency (move to tail of Map).
      this._compileFailures.delete(hashBigInt);
      this._compileFailures.set(hashBigInt, entry);
      this._compileFailureCacheHits += 1;
      return true;
    }

    /**
     * Inspect a sanitized negative compile-failure cache entry for `hash` without mutating
     * LRU ordering or hit counters. Returns `null` when not cached.
     *
     * @param {bigint | string | number | null | undefined} hash
     * @returns {{
     *   hashHex: string,
     *   stage: 'vertex' | 'fragment' | 'compute' | null,
     *   reason: string,
     *   count: number,
     *   firstSeenMs: number,
     *   lastSeenMs: number,
     * } | null}
     */
    getCompileFailure(hash) {
      if (hash === null || hash === undefined || hash === '') {
        return null;
      }
      const hashBigInt = normalizeShaderHash(hash);
      const entry = this._compileFailures.get(hashBigInt);
      if (entry === undefined) {
        return null;
      }
      return {
        hashHex: entry.hashHex,
        stage: entry.stage,
        reason: entry.reason,
        count: entry.count,
        firstSeenMs: entry.firstSeenMs,
        lastSeenMs: entry.lastSeenMs,
      };
    }

    /**
     * Remove a single shader hash from the negative compile-failure cache.
     *
     * @param {bigint | string | number | null | undefined} hash
     * @returns {boolean}
     */
    clearCompileFailure(hash) {
      if (hash === null || hash === undefined || hash === '') {
        return false;
      }
      const hashBigInt = normalizeShaderHash(hash);
      const existing = this._compileFailures.get(hashBigInt);
      if (existing === undefined) {
        return false;
      }
      this._compileFailures.delete(hashBigInt);
      this._evictResolveCacheForHash(hashBigInt, existing.hashHex);
      return true;
    }

    /**
     * Clear all entries from the negative compile-failure cache.
     * Returns the number of entries cleared.
     *
     * @returns {number}
     */
    clearCompileFailures() {
      const cleared = this._compileFailures.size;
      if (cleared > 0) {
        this._compileFailures.clear();
        this._clearResolveCache();
      }
      return cleared;
    }

    /**
     * Deterministically select a built-in fallback WGSL profile for `(stage, bindingSignature)`.
     *
     * @param {number | string} stage
     * @param {object | null | undefined} [bindingSignature]
     * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b'}
     */
    selectFallbackProfile(stage, bindingSignature) {
      return selectDeterministicFallbackProfile(stage, bindingSignature);
    }

    _storeResolveCache(
      stageCode,
      stageName,
      rawHash,
      hashBigInt,
      hashHex,
      result,
      source,
      profile,
      missingRow,
      categorizedRow,
      priorityRow
    ) {
      if (this._maxResolveCacheEntries <= 0) {
        return;
      }

      const packWgslBytes =
        source === 'pack' && result && typeof result.wgsl === 'string'
          ? result.wgsl.length * 2
          : 0;
      if (packWgslBytes > 0) {
        if (
          this._maxResolveCachePackWgslBytes <= 0 ||
          packWgslBytes > this._maxResolveCachePackWgslBytes
        ) {
          return;
        }
      }

      const stageCache = this._resolveCacheByStage[stageCode];
      const existing = stageCache.get(hashBigInt) || stageCache.get(hashHex) || stageCache.get(rawHash);
      if (existing !== undefined) {
        this._evictResolveCacheEntry(existing);
      }

      if (packWgslBytes > 0) {
        while (this._resolveCachePackWgslBytes + packWgslBytes > this._maxResolveCachePackWgslBytes) {
          let evictedPack = false;
          for (const candidate of this._resolveCacheEntries.keys()) {
            if (candidate.packWgslBytes > 0) {
              this._evictResolveCacheEntry(candidate);
              evictedPack = true;
              break;
            }
          }
          if (!evictedPack) {
            break;
          }
        }
      }

      if (this._resolveCacheEntries.size >= this._maxResolveCacheEntries) {
        const oldest = this._resolveCacheEntries.keys().next().value;
        if (oldest !== undefined) {
          this._evictResolveCacheEntry(oldest);
        }
      }

      const key1 = hashBigInt;
      const key2 = hashHex;
      const key3 = rawHash !== key1 && rawHash !== key2 ? rawHash : undefined;
      const numericKey =
        hashBigInt <= MAX_SAFE_INTEGER_BIGINT ? Number(hashBigInt) : undefined;
      const key4 = numericKey !== undefined && numericKey !== key3 ? numericKey : undefined;

      const entry = {
        stageCode,
        stageName,
        hashBigInt,
        hashHex,
        source,
        profile,
        result,
        packWgslBytes,
        missingRow,
        categorizedRow,
        priorityRow: priorityRow !== undefined ? priorityRow : null,
        telemetryGeneration: this._telemetryGeneration,
        key1,
        key2,
        key3,
        key4,
      };

      stageCache.set(key1, entry);
      stageCache.set(key2, entry);
      if (key3 !== undefined) {
        stageCache.set(key3, entry);
      }
      if (key4 !== undefined) {
        stageCache.set(key4, entry);
      }
      this._resolveCacheEntries.set(entry, entry);
      if (packWgslBytes > 0) {
        this._resolveCachePackWgslBytes += packWgslBytes;
      }
    }

    /**
     * Invalidate all memoized `resolve()` entries (e.g. after a shader pack reload).
     * Missing-shader telemetry and registered real shaders are preserved.
     */
    invalidateCache() {
      this._capturePackStamp();
      this._clearResolveCache();
    }

    /**
     * Alias for `invalidateCache()` to notify the router that the active shader pack reloaded.
     */
    onPackReload() {
      this.invalidateCache();
    }

    /**
     * Set the current frame index recorded on missing-shader `firstSeenFrame` / `lastSeenFrame`.
     *
     * @param {number} frameIndex
     * @returns {number}
     */
    setFrame(frameIndex) {
      this._currentFrame = normalizeFrameNumber(frameIndex);
      return this._currentFrame;
    }

    /**
     * Advance the current frame index by `delta` (default 1) and return the new frame index.
     *
     * @param {number} [delta=1]
     * @returns {number}
     */
    advanceFrame(delta = 1) {
      const step = normalizeFrameNumber(delta);
      this._currentFrame += step;
      return this._currentFrame;
    }

    /**
     * Return the current frame index tracked by the router.
     *
     * @returns {number}
     */
    getCurrentFrame() {
      return this._currentFrame;
    }

    /**
     * Return sanitized fast-path resolve cache telemetry.
     *
     * @returns {{
     *   hits: number,
     *   misses: number,
     *   entries: number,
     *   maxEntries: number,
     *   invalidations: number,
     *   hitRatio: number,
     * }}
     */
    getResolveCacheStats() {
      const total = this._resolveCacheHits + this._resolveCacheMisses;
      return {
        hits: this._resolveCacheHits,
        misses: this._resolveCacheMisses,
        entries: this._resolveCacheEntries.size,
        maxEntries: this._maxResolveCacheEntries,
        invalidations: this._resolveCacheInvalidations,
        hitRatio: total > 0 ? this._resolveCacheHits / total : 0,
      };
    }

    _registerInitialRealShaders(realShaders) {
      if (realShaders instanceof Map) {
        for (const [hash, meta] of realShaders.entries()) {
          if (meta && typeof meta === 'object' && meta.stage !== undefined) {
            this.registerRealShader(hash, meta.stage, meta);
          } else {
            this.registerRealShader(hash, '*', meta);
          }
        }
        return;
      }
      if (realShaders instanceof Set || Array.isArray(realShaders)) {
        for (const entry of realShaders) {
          if (
            typeof entry === 'string' ||
            typeof entry === 'bigint' ||
            typeof entry === 'number'
          ) {
            this.registerRealShader(entry, '*');
          } else if (entry && typeof entry === 'object' && entry.hash !== undefined) {
            this.registerRealShader(
              entry.hash,
              entry.stage !== undefined ? entry.stage : '*',
              entry
            );
          }
        }
        return;
      }
      if (typeof realShaders === 'object') {
        const keys = Object.keys(realShaders);
        for (let i = 0; i < keys.length; i += 1) {
          const hash = keys[i];
          const meta = realShaders[hash];
          if (meta && typeof meta === 'object' && meta.stage !== undefined) {
            this.registerRealShader(hash, meta.stage, meta);
          } else {
            this.registerRealShader(hash, '*', meta);
          }
        }
      }
    }

    _isConfiguredFallbackHash(hashBigInt) {
      const stages = ['vertex', 'fragment', 'compute'];
      for (let i = 0; i < stages.length; i += 1) {
        try {
          const fb = resolveFallbackHashForStage(this._fallbackHashes, stages[i]);
          if (fb !== null && fb === hashBigInt) {
            return true;
          }
        } catch (_err) {
          // Ignore malformed fallback hash entry during predicate check.
        }
      }
      return false;
    }

    _lookupRealShader(hashHex, stageName) {
      const exact = this._realShaders.get(`${hashHex}|${stageName}`);
      if (exact) {
        return exact;
      }
      return this._realShaders.get(`${hashHex}|*`) || null;
    }

    _recordBlockedOverride(hashHex, stageName, reason, nowMs) {
      this._blockedOverrideCount += 1;
      this._summaryVersion += 1;
      const key = `${hashHex}|${stageName}`;
      const existing = this._blockedOverrides.get(key);
      if (existing) {
        existing.count += 1;
        existing.lastReason = reason;
        existing.lastSeenMs = nowMs;
        return;
      }
      if (this._blockedOverrides.size >= this._maxCategorizedMissing) {
        this._blockedOverflowCount += 1;
        return;
      }
      this._blockedOverrides.set(key, {
        hashHex,
        stage: stageName,
        count: 1,
        lastReason: reason,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      });
    }

    /**
     * Register a real shader module (e.g. pack shader, menu shader, or Bink2 YUV shader)
     * so fallback resolution can never silently overwrite it with an opaque fallback.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | object | null} [stageOrMeta]
     * @param {string | {
     *   wgsl?: string | null,
     *   code?: string | null,
     *   entryPoint?: string | null,
     *   entry_point?: string | null,
     *   module?: object | null,
     *   source?: 'pack' | 'real',
     *   label?: string | null,
     *   stage?: number | string | null,
     * } | null} [maybeMeta]
     * @returns {{ hashHex: string, stage: string, hasWgsl: boolean, hasModule: boolean }}
     */
    registerRealShader(hash, stageOrMeta, maybeMeta) {
      const hashHex = formatShaderHash(normalizeShaderHash(hash));
      let stageArg = stageOrMeta;
      let meta = maybeMeta;

      if (
        maybeMeta === undefined &&
        stageOrMeta !== null &&
        typeof stageOrMeta === 'object'
      ) {
        meta = stageOrMeta;
        stageArg = stageOrMeta.stage !== undefined ? stageOrMeta.stage : '*';
      }

      const stageKey =
        stageArg === undefined || stageArg === null || stageArg === '*'
          ? '*'
          : normalizeStage(stageArg);

      let wgsl = null;
      let entryPoint = null;
      let moduleRef = null;
      let source = 'real';
      let label = null;

      if (typeof meta === 'string') {
        if (meta.trim().length > 0) {
          wgsl = meta;
        }
      } else if (meta && typeof meta === 'object') {
        if (typeof meta.wgsl === 'string' && meta.wgsl.trim().length > 0) {
          wgsl = meta.wgsl;
        } else if (typeof meta.code === 'string' && meta.code.trim().length > 0) {
          wgsl = meta.code;
        }
        if (typeof meta.entryPoint === 'string' && meta.entryPoint.length > 0) {
          entryPoint = meta.entryPoint;
        } else if (typeof meta.entry_point === 'string' && meta.entry_point.length > 0) {
          entryPoint = meta.entry_point;
        }
        if (meta.module !== undefined && meta.module !== null) {
          moduleRef = meta.module;
        }
        if (meta.source === 'pack' || meta.source === 'real') {
          source = meta.source;
        }
        if (typeof meta.label === 'string' && meta.label.length > 0) {
          label = meta.label;
        }
      }

      const record = {
        hashHex,
        stage: stageKey,
        wgsl,
        entryPoint,
        module: moduleRef,
        source,
        label,
        registeredAtMs: this._now(),
      };
      const realKey = `${hashHex}|${stageKey}`;
      if (this._realShaders.has(realKey)) {
        this._realShaders.delete(realKey);
      } else if (this._realShaders.size >= this._maxRealShaders) {
        const oldestKey = this._realShaders.keys().next().value;
        if (oldestKey !== undefined) {
          this._realShaders.delete(oldestKey);
        }
      }
      this._realShaders.set(realKey, record);
      this._summaryVersion += 1;
      this._clearResolveCache();

      return {
        hashHex,
        stage: stageKey,
        hasWgsl: wgsl !== null,
        hasModule: moduleRef !== null,
      };
    }

    /**
     * Unregister a previously registered real shader for `(hash, stage)`.
     * When `stage` is omitted, removes all stage entries registered for `hash`.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @returns {boolean}
     */
    unregisterRealShader(hash, stage) {
      const hashHex = formatShaderHash(normalizeShaderHash(hash));
      if (stage === undefined || stage === null) {
        let removed = false;
        const prefix = `${hashHex}|`;
        for (const key of Array.from(this._realShaders.keys())) {
          if (key.startsWith(prefix)) {
            this._realShaders.delete(key);
            removed = true;
          }
        }
        if (removed) {
          this._summaryVersion += 1;
          this._clearResolveCache();
        }
        return removed;
      }
      const stageKey = stage === '*' ? '*' : normalizeStage(stage);
      const removed = this._realShaders.delete(`${hashHex}|${stageKey}`);
      if (removed) {
        this._summaryVersion += 1;
        this._clearResolveCache();
      }
      return removed;
    }

    /**
     * Clear all explicitly registered real shaders.
     */
    clearRealShaders() {
      if (this._realShaders.size > 0) {
        this._realShaders.clear();
        this._summaryVersion += 1;
        this._clearResolveCache();
      }
    }

    /**
     * Return true if a real shader module exists for `(hash, stage)` either in the
     * explicit real-shader registry or as a non-fallback entry in the active pack loader.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @returns {boolean}
     */
    hasRealShader(hash, stage) {
      const hashBigInt = normalizeShaderHash(hash);
      const hashHex = formatShaderHash(hashBigInt);

      if (stage === undefined || stage === null || stage === '*') {
        if (this._realShaders.has(`${hashHex}|*`)) {
          return true;
        }
        const prefix = `${hashHex}|`;
        for (const key of this._realShaders.keys()) {
          if (key.startsWith(prefix)) {
            return true;
          }
        }
      } else {
        const stageName = normalizeStage(stage);
        if (this._lookupRealShader(hashHex, stageName) !== null) {
          return true;
        }
      }

      if (!this._isConfiguredFallbackHash(hashBigInt)) {
        const packHit = lookupPackSource(this._packLoader, hashBigInt, hashHex);
        if (packHit || isPendingRangedPackShader(this._packLoader, hashBigInt, hashHex)) {
          return true;
        }
      }
      return false;
    }

    /**
     * Guard predicate returning `false` whenever a real shader exists for `(hash, stage)`
     * or is supplied in `context`, ensuring callers never silently overwrite real menu/video
     * shaders with fallback shaders.
     *
     * @param {bigint | string | number} hash
     * @param {number | string | null} [stage]
     * @param {object | null} [context]
     * @returns {boolean}
     */
    canFallbackOverride(hash, stage, context) {
      if (this.hasRealShader(hash, stage)) {
        return false;
      }
      if (context && typeof context === 'object') {
        const existingShader =
          context.existingShader && typeof context.existingShader === 'object'
            ? context.existingShader
            : null;
        const src =
          context.existingSource ||
          context.source ||
          (existingShader && existingShader.source) ||
          null;
        if (
          context.isRealShader === true ||
          src === 'pack' ||
          src === 'real' ||
          (existingShader && existingShader.isRealShader === true)
        ) {
          return false;
        }
      }
      return true;
    }

    /**
     * Return JSON-safe real-shader override guard telemetry (zero raw WGSL or host paths).
     *
     * @param {{ limit?: number }} [options]
     * @returns {{
     *   registeredRealShaders: number,
     *   blockedOverrideCount: number,
     *   overflowBlockedCount: number,
     *   blockedShaders: Array<{
     *     hashHex: string,
     *     stage: string,
     *     count: number,
     *     lastReason: string,
     *     firstSeenMs: number,
     *     lastSeenMs: number,
     *   }>
     * }}
     */
    getRealShaderGuardStats(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      let limit = opts.limit;
      if (limit === undefined || limit === null) {
        limit = this._blockedOverrides.size;
      }
      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }

      const blockedShaders = Array.from(this._blockedOverrides.values(), (row) => ({
        hashHex: row.hashHex,
        stage: row.stage,
        count: row.count,
        lastReason: row.lastReason,
        firstSeenMs: row.firstSeenMs,
        lastSeenMs: row.lastSeenMs,
      }));

      blockedShaders.sort(
        (a, b) =>
          b.count - a.count ||
          compareLex(a.hashHex, b.hashHex) ||
          compareLex(a.stage, b.stage)
      );

      return {
        registeredRealShaders: this._realShaders.size,
        blockedOverrideCount: this._blockedOverrideCount,
        overflowBlockedCount: this._blockedOverflowCount,
        blockedShaders: blockedShaders.slice(0, limit),
      };
    }

    /**
     * Return a sanitized map of available built-in fallback WGSL profiles.
     * @returns {Record<string, { vertex: string, fragment: string, compute: string }>}
     */
    getBuiltInFallbackProfiles() {
      return getBuiltInFallbackProfiles();
    }

    /**
     * Replace the pack loader used for exact / stage-fallback lookups and invalidate
     * the resolved-hash memoization cache.
     * @param {object | Map | null | undefined} packLoader
     */
    setPackLoader(packLoader) {
      this._packLoader = packLoader !== undefined ? packLoader : null;
      this.invalidateCache();
    }

    /**
     * Replace per-stage S9 fallback hashes (`vertex` / `fragment` / `compute`) and
     * invalidate the resolved-hash memoization cache.
     * @param {Record<string, bigint | string | number> | null | undefined} fallbackHashes
     */
    setFallbackHashes(fallbackHashes) {
      this._fallbackHashes =
        fallbackHashes && typeof fallbackHashes === 'object' ? { ...fallbackHashes } : {};
      this.invalidateCache();
    }

    /**
     * Record a completed asynchronous ranged-pack shader hit so `packHits`, `totalLookups`,
     * and `hitRatePct` accurately reflect ranged HTTP 206 shader resolutions.
     *
     * @param {bigint | string | number | null | undefined} [hash]
     * @param {number | string | null | undefined} [_stage]
     */
    recordPackHit(hash, _stage) {
      if (hash !== undefined && hash !== null && hash !== '') {
        try {
          normalizeShaderHash(hash);
        } catch (_err) {
          // Non-fatal: still count the pack hit even if caller passed an opaque label.
        }
      }
      this._stats.totalLookups += 1;
      this._stats.packHits += 1;
    }

    /**
     * Resolve a requested shader hash for a stage:
     * 1) exact pack hit (never silently overwritten by fallbacks),
     * 2) registered / caller-supplied real shader guard (never silently overwritten by fallbacks),
     * 3) stage-matched S9 fallback hash in pack,
     * 4) built-in WGSL.
     *
     * Hot repeat calls hit the per-stage resolved-hash memoization cache with zero
     * temporary object allocations and O(1) missing-hash counter updates.
     *
     * @param {bigint | string | number} hash
     * @param {number | string} stage
     * @param {{
     *   profile?: string | null,
     *   frame?: number | null,
     *   forceFallback?: boolean,
     *   isRealShader?: boolean,
     *   existingSource?: string | null,
     *   existingShader?: {
     *     wgsl?: string | null,
     *     code?: string | null,
     *     entryPoint?: string | null,
     *     module?: object | null,
     *     source?: string | null,
     *     isRealShader?: boolean,
     *   } | null,
     *   reason?: string | null,
     *   missingReason?: string | null,
     * }} [options]
     * @returns {{
     *   wgsl: string | null,
     *   source: 'pack' | 'real' | 'fallback' | 'builtin',
     *   requestedHash: string,
     *   resolvedHash: string | null,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   entryPoint: string | null,
     *   profile?: 'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | 'splash_rect_24b',
     *   fallbackBlocked?: boolean,
     *   guardReason?: string,
     *   module?: object,
     *   label?: string,
     * }}
     */
    resolve(hash, stage, options) {
      const stageCode = normalizeStageCode(stage);
      const stageName = STAGE_NAMES[stageCode];

      if (this._packLoader !== null) {
        this._syncPackCacheVersion();
      }

      const opts = options !== undefined && options !== null && typeof options === 'object' ? options : null;
      const hasInterstageVertexOpts =
        stageCode === SHADER_STAGE.VERTEX &&
        opts !== null &&
        (opts.interstageLocations !== undefined ||
          opts.psWgsl !== undefined ||
          opts.interstageLocationCount !== undefined);

      const hasSpecialOpts =
        opts !== null &&
        (opts.forceFallback === true ||
          opts.isRealShader === true ||
          opts.rangedPending === true ||
          opts.existingSource !== undefined ||
          opts.existingShader !== undefined ||
          opts.reason !== undefined ||
          opts.missingReason !== undefined ||
          hasInterstageVertexOpts);

      // W204/W304 Fast Path: zero-alloc O(1) BigInt/numeric/canonical-hex routing (no hex formatting).
      let requested;
      let hasRequested = false;
      if (!hasSpecialOpts && this._maxResolveCacheEntries > 0) {
        const stageCache = this._resolveCacheByStage[stageCode];
        let cached = stageCache.get(hash);
        if (cached === undefined) {
          requested = normalizeShaderHash(hash);
          hasRequested = true;
          cached = stageCache.get(requested);
          if (
            cached !== undefined &&
            cached.key3 === undefined &&
            hash !== cached.key1 &&
            hash !== cached.key2 &&
            hash !== cached.key4
          ) {
            cached.key3 = hash;
            stageCache.set(hash, cached);
          }
        }
        if (cached !== undefined) {
          if (cached.source === 'pack') {
            this._touchResolveCacheEntry(cached);
            this._stats.totalLookups += 1;
            this._stats.packHits += 1;
            this._resolveCacheHits += 1;
            this._fastPathHits += 1;
            return cached.result;
          }
          if (cached.source === 'fallback') {
            const frame =
              opts !== null && opts.frame !== undefined && opts.frame !== null
                ? normalizeFrameNumber(opts.frame)
                : this._currentFrame;
            this._touchResolveCacheEntry(cached);
            this._stats.totalLookups += 1;
            this._stats.fallbackHits += 1;
            this._resolveCacheHits += 1;
            this._fastPathHits += 1;
            this._recordCachedMissHit(cached, this._now(), frame);
            return cached.result;
          }
          if (cached.source === 'builtin') {
            const profileName =
              opts !== null && opts.profile !== undefined && opts.profile !== null
                ? normalizeFallbackProfile(opts.profile)
                : opts !== null && opts.bindingSignature !== undefined && opts.bindingSignature !== null
                  ? selectDeterministicFallbackProfile(stageCode, opts.bindingSignature)
                  : this._defaultProfile;
            if (cached.profile === profileName) {
              const frame =
                opts !== null && opts.frame !== undefined && opts.frame !== null
                  ? normalizeFrameNumber(opts.frame)
                  : this._currentFrame;
              this._touchResolveCacheEntry(cached);
              this._stats.totalLookups += 1;
              this._stats.builtinHits += 1;
              this._resolveCacheHits += 1;
              this._fastPathHits += 1;
              this._recordCachedMissHit(cached, this._now(), frame);
              return cached.result;
            }
          }
        }
      }

      if (!hasRequested) {
        requested = normalizeShaderHash(hash);
      }
      const frame =
        opts !== null && opts.frame !== undefined && opts.frame !== null
          ? normalizeFrameNumber(opts.frame)
          : this._currentFrame;
      const profileName =
        opts !== null && opts.profile !== undefined && opts.profile !== null
          ? normalizeFallbackProfile(opts.profile)
          : opts !== null && opts.bindingSignature !== undefined && opts.bindingSignature !== null
            ? selectDeterministicFallbackProfile(stageCode, opts.bindingSignature)
            : this._defaultProfile;

      if (!hasSpecialOpts) {
        this._resolveCacheMisses += 1;
      }

      this._stats.totalLookups += 1;

      // W268: Negative compile-failure cache — skip retrying known-broken hashes and
      // route straight to the deterministic fallback shader without spamming Tint errors.
      // Keyed by u64 BigInt (W304) so repeated BigInt/numeric lookups avoid hex string formatting.
      const compileFailureEntry =
        this._compileFailures.size > 0 ? this._compileFailures.get(requested) : undefined;
      if (compileFailureEntry !== undefined) {
        this._compileFailures.delete(requested);
        this._compileFailures.set(requested, compileFailureEntry);
        this._compileFailureCacheHits += 1;

        const requestedHex = compileFailureEntry.hashHex;
        const nowMs = this._now();
        this._trackMissing(requested, requestedHex, stageCode, stageName, nowMs, frame);
        this._trackMissingPriorityHash(requested, requestedHex, stageCode, stageName, nowMs);
        const failReason =
          opts !== null && (opts.reason !== undefined || opts.missingReason !== undefined)
            ? normalizeMissingReason(opts.reason !== undefined ? opts.reason : opts.missingReason)
            : MISSING_SHADER_REASONS.COMPILE_FAILED;
        this._recordCategorizedMissingRow(requestedHex, stageName, failReason, nowMs);

        const fallbackHash = resolveFallbackHashForStage(this._fallbackHashes, stageName);
        if (fallbackHash !== null && !this._compileFailures.has(fallbackHash)) {
          const resolvedHex = formatShaderHash(fallbackHash);
          const fallbackHit = lookupPackSource(this._packLoader, fallbackHash, resolvedHex);
          if (fallbackHit) {
            this._stats.fallbackHits += 1;
            return Object.freeze({
              wgsl: fallbackHit.wgsl,
              source: 'fallback',
              requestedHash: requestedHex,
              resolvedHash: resolvedHex,
              stage: stageName,
              entryPoint: fallbackHit.entryPoint || extractEntryPoint(fallbackHit.wgsl, stageName),
              compileFailureCached: true,
              compileFailureReason: compileFailureEntry.reason,
            });
          }
        }

        const profileTemplates = BUILTIN_FALLBACK_PROFILES[profileName] || BUILTIN_FALLBACK_WGSL;
        const builtin = hasInterstageVertexOpts
          ? buildInterstageFallbackVertexWgsl({
              locations: opts.interstageLocations,
              psWgsl: opts.psWgsl,
              locationCount: opts.interstageLocationCount,
            })
          : profileTemplates[stageName] || BUILTIN_FALLBACK_WGSL[stageName];
        this._stats.builtinHits += 1;
        return Object.freeze({
          wgsl: builtin,
          source: 'builtin',
          requestedHash: requestedHex,
          resolvedHash: null,
          stage: stageName,
          entryPoint: extractEntryPoint(builtin, stageName),
          profile: profileName,
          compileFailureCached: true,
          compileFailureReason: compileFailureEntry.reason,
        });
      }

      const requestedHex = formatShaderHash(requested);

      const exact = lookupPackSource(this._packLoader, requested, requestedHex);
      if (exact) {
        this._stats.packHits += 1;
        if (opts !== null && opts.forceFallback === true && !this._isConfiguredFallbackHash(requested)) {
          const nowMs = this._now();
          this._recordBlockedOverride(requestedHex, stageName, 'pack_shader_protected', nowMs);
          return {
            wgsl: exact.wgsl,
            source: 'pack',
            requestedHash: requestedHex,
            resolvedHash: requestedHex,
            stage: stageName,
            entryPoint: exact.entryPoint || extractEntryPoint(exact.wgsl, stageName),
            fallbackBlocked: true,
            guardReason: 'pack_shader_protected',
          };
        }
        const result = Object.freeze({
          wgsl: exact.wgsl,
          source: 'pack',
          requestedHash: requestedHex,
          resolvedHash: requestedHex,
          stage: stageName,
          entryPoint: exact.entryPoint || extractEntryPoint(exact.wgsl, stageName),
        });
        if (!hasSpecialOpts) {
          this._storeResolveCache(
            stageCode,
            stageName,
            hash,
            requested,
            requestedHex,
            result,
            'pack',
            null,
            null,
            null,
            null
          );
        }
        return result;
      }

      const nowMs = this._now();

      // W179: Real-shader override guard — never silently replace a registered real shader
      // (e.g. menu or Bink2 YUV shader) with an S9 or built-in opaque fallback.
      const registeredReal = this._lookupRealShader(requestedHex, stageName);
      if (registeredReal) {
        this._recordBlockedOverride(requestedHex, stageName, 'real_shader_registered', nowMs);
        if (registeredReal.source === 'pack') {
          this._stats.packHits += 1;
        }
        const guarded = {
          wgsl: registeredReal.wgsl,
          source: registeredReal.source,
          requestedHash: requestedHex,
          resolvedHash: requestedHex,
          stage: stageName,
          entryPoint:
            registeredReal.entryPoint ||
            (registeredReal.wgsl
              ? extractEntryPoint(registeredReal.wgsl, stageName)
              : DEFAULT_ENTRY_POINTS[stageName] || null),
          fallbackBlocked: true,
          guardReason: 'real_shader_registered',
        };
        if (registeredReal.module !== null) {
          guarded.module = registeredReal.module;
        }
        if (registeredReal.label !== null) {
          guarded.label = registeredReal.label;
        }
        return guarded;
      }

      // W179: Caller-indicated existing real shader guard.
      if (opts !== null) {
        const existingShader =
          opts.existingShader && typeof opts.existingShader === 'object'
            ? opts.existingShader
            : null;
        const existingSource =
          opts.existingSource || (existingShader && existingShader.source) || null;
        const callerClaimsReal =
          opts.isRealShader === true ||
          existingSource === 'pack' ||
          existingSource === 'real' ||
          (existingShader !== null && existingShader.isRealShader === true);

        if (callerClaimsReal) {
          this._recordBlockedOverride(requestedHex, stageName, 'existing_real_shader', nowMs);
          const existingWgsl =
            existingShader && typeof existingShader.wgsl === 'string' && existingShader.wgsl.trim().length > 0
              ? existingShader.wgsl
              : existingShader && typeof existingShader.code === 'string' && existingShader.code.trim().length > 0
                ? existingShader.code
                : null;
          const resolvedSource = existingSource === 'pack' ? 'pack' : 'real';
          if (resolvedSource === 'pack') {
            this._stats.packHits += 1;
          }
          const guarded = {
            wgsl: existingWgsl,
            source: resolvedSource,
            requestedHash: requestedHex,
            resolvedHash: requestedHex,
            stage: stageName,
            entryPoint:
              (existingShader && typeof existingShader.entryPoint === 'string' && existingShader.entryPoint) ||
              (existingWgsl
                ? extractEntryPoint(existingWgsl, stageName)
                : DEFAULT_ENTRY_POINTS[stageName] || null),
            fallbackBlocked: true,
            guardReason: 'existing_real_shader',
          };
          if (existingShader && existingShader.module !== undefined && existingShader.module !== null) {
            guarded.module = existingShader.module;
          }
          return guarded;
        }
      }

      const pendingRanged =
        (opts !== null && opts.rangedPending === true) ||
        isPendingRangedPackShader(this._packLoader, requested, requestedHex);

      let missingRow = null;
      let priorityRow = null;
      let categorizedRow = null;
      if (pendingRanged) {
        // Shader exists in the ranged pack's TOC and is awaiting HTTP 206 fetch/compile.
        // Do not record it as missing (`not_in_pack`) or inflate `builtinHits`/`totalLookups`;
        // the real outcome will be recorded once ranged fetch/compile settles.
        this._stats.totalLookups = Math.max(0, this._stats.totalLookups - 1);
        this._rangedPendingHits += 1;
      } else {
        // Exact miss — record Frame-1, categorized, and W304 priority-hash telemetry in O(1).
        missingRow = this._trackMissing(requested, requestedHex, stageCode, stageName, nowMs, frame);
        priorityRow = this._trackMissingPriorityHash(requested, requestedHex, stageCode, stageName, nowMs);
        const missReason =
          opts !== null && (opts.reason !== undefined || opts.missingReason !== undefined)
            ? normalizeMissingReason(opts.reason !== undefined ? opts.reason : opts.missingReason)
            : MISSING_SHADER_REASONS.NOT_IN_PACK;
        categorizedRow = this._recordCategorizedMissingRow(requestedHex, stageName, missReason, nowMs);
      }

      const canCacheMiss = !hasSpecialOpts && !pendingRanged;

      const fallbackHash = resolveFallbackHashForStage(this._fallbackHashes, stageName);
      if (fallbackHash !== null && !this._compileFailures.has(fallbackHash)) {
        const resolvedHex = formatShaderHash(fallbackHash);
        const fallbackHit = lookupPackSource(this._packLoader, fallbackHash, resolvedHex);
        if (fallbackHit) {
          if (!pendingRanged) {
            this._stats.fallbackHits += 1;
          }
          const resultObj = {
            wgsl: fallbackHit.wgsl,
            source: 'fallback',
            requestedHash: requestedHex,
            resolvedHash: resolvedHex,
            stage: stageName,
            entryPoint: fallbackHit.entryPoint || extractEntryPoint(fallbackHit.wgsl, stageName),
          };
          if (pendingRanged) {
            resultObj.rangedPending = true;
          }
          const result = Object.freeze(resultObj);
          if (canCacheMiss) {
            this._storeResolveCache(
              stageCode,
              stageName,
              hash,
              requested,
              requestedHex,
              result,
              'fallback',
              null,
              missingRow,
              categorizedRow,
              priorityRow
            );
          }
          return result;
        }
      }

      const profileTemplates = BUILTIN_FALLBACK_PROFILES[profileName] || BUILTIN_FALLBACK_WGSL;
      const builtin = hasInterstageVertexOpts
        ? buildInterstageFallbackVertexWgsl({
            locations: opts.interstageLocations,
            psWgsl: opts.psWgsl,
            locationCount: opts.interstageLocationCount,
          })
        : profileTemplates[stageName] || BUILTIN_FALLBACK_WGSL[stageName];
      if (!pendingRanged) {
        this._stats.builtinHits += 1;
      }
      const resultObj = {
        wgsl: builtin,
        source: 'builtin',
        requestedHash: requestedHex,
        resolvedHash: null,
        stage: stageName,
        entryPoint: extractEntryPoint(builtin, stageName),
        profile: profileName,
      };
      if (pendingRanged) {
        resultObj.rangedPending = true;
      }
      const result = Object.freeze(resultObj);
      if (canCacheMiss) {
        this._storeResolveCache(
          stageCode,
          stageName,
          hash,
          requested,
          requestedHex,
          result,
          'builtin',
          profileName,
          missingRow,
          categorizedRow,
          priorityRow
        );
      }
      return result;
    }

    _recordCachedMissHit(cached, nowMs, frame) {
      if (cached.telemetryGeneration === this._telemetryGeneration) {
        const row = cached.missingRow;
        if (row !== null) {
          row.hitCount += 1;
          row.lastSeenMs = nowMs;
          if (frame < row.firstSeenFrame) {
            row.firstSeenFrame = frame;
          }
          if (frame > row.lastSeenFrame) {
            row.lastSeenFrame = frame;
          }
        } else {
          this._overflowMissingCount += 1;
        }

        const pRow = cached.priorityRow;
        if (pRow !== null && !pRow.evicted) {
          pRow.hitCount += 1;
          pRow.stageMask |= 1 << cached.stageCode;
          pRow.lastStage = cached.stageName;
          pRow.lastSeenMs = nowMs;
          this._missingPrioritySeq += 1;
          pRow.lastSeenSeq = this._missingPrioritySeq;
        } else if (this._maxMissingHashes > 0) {
          cached.priorityRow = this._trackMissingPriorityHash(
            cached.hashBigInt,
            cached.hashHex,
            cached.stageCode,
            cached.stageName,
            nowMs
          );
        }

        this._categorizedTotals.total += 1;
        this._categorizedTotals.byReason.not_in_pack += 1;
        this._categorizedTotals.byStage[cached.stageName] += 1;
        this._summaryVersion += 1;
        const catRow = cached.categorizedRow;
        if (catRow !== null) {
          catRow.hitCount += 1;
          catRow.reasons.not_in_pack += 1;
          catRow.lastReason = MISSING_SHADER_REASONS.NOT_IN_PACK;
          catRow.lastSeenMs = nowMs;
        } else {
          this._categorizedOverflowCount += 1;
        }
        return;
      }

      // Telemetry was reset while the resolved shader stayed memoized — re-bind O(1) rows cleanly.
      cached.missingRow = this._trackMissing(
        cached.hashBigInt,
        cached.hashHex,
        cached.stageCode,
        cached.stageName,
        nowMs,
        frame
      );
      cached.priorityRow = this._trackMissingPriorityHash(
        cached.hashBigInt,
        cached.hashHex,
        cached.stageCode,
        cached.stageName,
        nowMs
      );
      cached.categorizedRow = this._recordCategorizedMissingRow(
        cached.hashHex,
        cached.stageName,
        MISSING_SHADER_REASONS.NOT_IN_PACK,
        nowMs
      );
      cached.telemetryGeneration = this._telemetryGeneration;
    }

    _trackMissingPriorityHash(hashBigInt, hashHex, stageCode, stageName, nowMs) {
      if (this._maxMissingHashes <= 0) {
        return null;
      }

      const existing = this._missingPriorityHashes.get(hashBigInt);
      if (existing !== undefined) {
        existing.hitCount += 1;
        if (stageCode > 0) {
          existing.stageMask |= 1 << stageCode;
          existing.lastStage = stageName;
        }
        existing.lastSeenMs = nowMs;
        this._missingPrioritySeq += 1;
        existing.lastSeenSeq = this._missingPrioritySeq;
        return existing;
      }

      if (this._missingPriorityHashes.size >= this._maxMissingHashes) {
        let victimKey;
        let victimRow = null;
        for (const [k, row] of this._missingPriorityHashes.entries()) {
          if (
            victimRow === null ||
            row.hitCount < victimRow.hitCount ||
            (row.hitCount === victimRow.hitCount && row.lastSeenSeq < victimRow.lastSeenSeq)
          ) {
            victimKey = k;
            victimRow = row;
          }
        }
        if (victimRow !== null) {
          victimRow.evicted = true;
          this._missingPriorityHashes.delete(victimKey);
          this._missingPriorityEvictions += 1;
        }
      }

      this._missingPrioritySeq += 1;
      const row = {
        hashBigInt,
        hashHex,
        hitCount: 1,
        stageMask: stageCode > 0 ? 1 << stageCode : 0,
        lastStage: stageName,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
        lastSeenSeq: this._missingPrioritySeq,
        evicted: false,
      };
      this._missingPriorityHashes.set(hashBigInt, row);
      return row;
    }

    _trackMissing(hashBigInt, hashHex, stageCode, stageName, nowMs, frame) {
      const stageMap = this._missingByStage[stageCode];
      const existing = stageMap.get(hashBigInt);
      if (existing !== undefined) {
        existing.hitCount += 1;
        existing.lastSeenMs = nowMs;
        if (frame < existing.firstSeenFrame) {
          existing.firstSeenFrame = frame;
        }
        if (frame > existing.lastSeenFrame) {
          existing.lastSeenFrame = frame;
        }
        return existing;
      }

      if (this._missingCount >= this._maxTrackedMissing) {
        this._overflowMissingCount += 1;
        return null;
      }

      const row = {
        hash: hashHex,
        stage: stageName,
        hitCount: 1,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
        firstSeenFrame: frame,
        lastSeenFrame: frame,
      };
      stageMap.set(hashBigInt, row);
      this._missingCount += 1;
      return row;
    }

    _recordCategorizedMissingRow(hashHex, stageName, reason, nowMs) {
      this._categorizedTotals.total += 1;
      this._categorizedTotals.byReason[reason] += 1;
      this._categorizedTotals.byStage[stageName] += 1;
      this._summaryVersion += 1;

      const key = `${hashHex}|${stageName}`;
      const existing = this._categorizedMissing.get(key);
      if (existing) {
        existing.hitCount += 1;
        existing.reasons[reason] += 1;
        existing.lastReason = reason;
        existing.lastSeenMs = nowMs;
        return existing;
      }

      if (this._categorizedMissing.size >= this._maxCategorizedMissing) {
        this._categorizedOverflowCount += 1;
        return null;
      }

      const reasons = createEmptyReasonCounts();
      reasons[reason] = 1;
      const row = {
        hashHex,
        stage: stageName,
        hitCount: 1,
        reasons,
        lastReason: reason,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
      };
      this._categorizedMissing.set(key, row);
      return row;
    }

    _recordCategorizedMissing(hashHex, stageName, reason, nowMs) {
      const row = this._recordCategorizedMissingRow(hashHex, stageName, reason, nowMs);
      if (row === null) {
        return {
          hashHex,
          stage: stageName,
          reason,
          hitCount: 0,
          tracked: false,
          overflowed: true,
        };
      }
      return {
        hashHex,
        stage: stageName,
        reason,
        hitCount: row.hitCount,
        tracked: true,
        overflowed: false,
      };
    }

    /**
     * Record a categorized missing or failed shader event for live Chrome triage.
     * Supports `'not_in_pack'`, `'reflection_failed'`, `'compile_failed'`, and `'other'`.
     * Bounded to `maxCategorizedMissingHashes` (default 8192) unique `(hashHex, stage)` entries.
     *
     * @param {bigint | string | number} hashHex
     * @param {number | string} stage
     * @param {string} [reason='other']
     * @returns {{
     *   hashHex: string,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   reason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *   hitCount: number,
     *   tracked: boolean,
     *   overflowed: boolean,
     * }}
     */
    recordMissingShader(hashHex, stage, reason = MISSING_SHADER_REASONS.OTHER) {
      const hashBigInt = normalizeShaderHash(hashHex);
      const canonicalHash = formatShaderHash(hashBigInt);
      const stageCode = normalizeStageCode(stage);
      const stageName = STAGE_NAMES[stageCode];
      const canonicalReason = normalizeMissingReason(reason);
      const nowMs = this._now();
      this._trackMissingPriorityHash(hashBigInt, canonicalHash, stageCode, stageName, nowMs);
      if (canonicalReason === MISSING_SHADER_REASONS.COMPILE_FAILED) {
        this._recordCompileFailureEntry(
          hashBigInt,
          canonicalHash,
          stageName,
          canonicalReason,
          nowMs
        );
      }
      return this._recordCategorizedMissing(
        canonicalHash,
        stageName,
        canonicalReason,
        nowMs
      );
    }

    /**
     * Export a JSON-safe summary of categorized missing shaders:
     * total & unique counts per category/stage, overflow counter, real-shader guard stats,
     * and top hit-count hashes sorted descending (deterministic hashHex/stage tie-breaking).
     *
     * @param {number | { limit?: number | null, reason?: string | null, stage?: number | string | null } | null} [options]
     * @returns {{
     *   totalMissingEvents: number,
     *   uniqueMissingHashes: number,
     *   maxTrackedHashes: number,
     *   overflowCount: number,
     *   countsByReason: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *   countsByStage: { vertex: number, fragment: number, compute: number },
     *   uniqueCountsByReason: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *   uniqueCountsByStage: { vertex: number, fragment: number, compute: number },
     *   realShaderGuard: {
     *     registeredRealShaders: number,
     *     blockedOverrideCount: number,
     *     overflowBlockedCount: number,
     *   },
     *   topMissing: Array<{
     *     hashHex: string,
     *     stage: 'vertex' | 'fragment' | 'compute',
     *     reason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *     lastReason: 'not_in_pack' | 'reflection_failed' | 'compile_failed' | 'other',
     *     hitCount: number,
     *     reasons: { not_in_pack: number, reflection_failed: number, compile_failed: number, other: number },
     *     firstSeenMs: number,
     *     lastSeenMs: number,
     *   }>
     * }}
     */
    getMissingShaderSummary(options) {
      let limit = DEFAULT_MISSING_SUMMARY_TOP_LIMIT;
      let filterReason = null;
      let filterStage = null;

      if (typeof options === 'number') {
        limit = options;
      } else if (options !== undefined && options !== null) {
        if (typeof options !== 'object') {
          throw new RangeError(`options must be an object or non-negative integer limit, got ${options}`);
        }
        if (options.limit !== undefined && options.limit !== null) {
          limit = options.limit;
        }
        if (options.reason !== undefined && options.reason !== null) {
          filterReason = normalizeMissingReason(options.reason);
        }
        if (options.stage !== undefined && options.stage !== null) {
          filterStage = normalizeStage(options.stage);
        }
      }

      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }

      if (this._summaryCacheVersion !== this._summaryVersion) {
        this._summaryCache.clear();
        this._summaryUniqueCountsCache = null;
        this._summaryCacheVersion = this._summaryVersion;
      }

      const cacheKey = `${filterReason || '*'}|${filterStage || '*'}|${limit}`;
      let cached = this._summaryCache.get(cacheKey);
      if (cached === undefined) {
        let uniqueCounts = this._summaryUniqueCountsCache;
        let topMissing;
        if (limit === 0 && uniqueCounts !== null) {
          topMissing = [];
        } else {
          const computeUnique = uniqueCounts === null;
          const uniqueCountsByReason = computeUnique
            ? createEmptyReasonCounts()
            : uniqueCounts.byReason;
          const uniqueCountsByStage = computeUnique
            ? createEmptyStageCounts()
            : uniqueCounts.byStage;
          const filteredRows = limit === 0 ? null : [];

          for (const row of this._categorizedMissing.values()) {
            if (computeUnique) {
              uniqueCountsByStage[row.stage] += 1;
              if (row.reasons.not_in_pack > 0) uniqueCountsByReason.not_in_pack += 1;
              if (row.reasons.reflection_failed > 0) uniqueCountsByReason.reflection_failed += 1;
              if (row.reasons.compile_failed > 0) uniqueCountsByReason.compile_failed += 1;
              if (row.reasons.other > 0) uniqueCountsByReason.other += 1;
            }

            if (filteredRows === null) {
              continue;
            }
            if (filterStage !== null && row.stage !== filterStage) {
              continue;
            }
            if (filterReason !== null && !(row.reasons[filterReason] > 0)) {
              continue;
            }

            filteredRows.push({
              hashHex: row.hashHex,
              stage: row.stage,
              reason: filterReason !== null ? filterReason : getDominantMissingReason(row),
              lastReason: row.lastReason,
              hitCount: row.hitCount,
              reasons: {
                not_in_pack: row.reasons.not_in_pack,
                reflection_failed: row.reasons.reflection_failed,
                compile_failed: row.reasons.compile_failed,
                other: row.reasons.other,
              },
              firstSeenMs: row.firstSeenMs,
              lastSeenMs: row.lastSeenMs,
            });
          }

          if (computeUnique) {
            uniqueCounts = { byReason: uniqueCountsByReason, byStage: uniqueCountsByStage };
            this._summaryUniqueCountsCache = uniqueCounts;
          }

          if (filteredRows === null || filteredRows.length === 0) {
            topMissing = [];
          } else {
            filteredRows.sort((a, b) => {
              if (filterReason !== null) {
                const diffReason = b.reasons[filterReason] - a.reasons[filterReason];
                if (diffReason !== 0) {
                  return diffReason;
                }
              }
              return (
                b.hitCount - a.hitCount ||
                compareLex(a.hashHex, b.hashHex) ||
                compareLex(a.stage, b.stage)
              );
            });
            topMissing = filteredRows.length > limit ? filteredRows.slice(0, limit) : filteredRows;
          }
        }

        cached = {
          totalMissingEvents: this._categorizedTotals.total,
          uniqueMissingHashes: this._categorizedMissing.size,
          maxTrackedHashes: this._maxCategorizedMissing,
          overflowCount: this._categorizedOverflowCount,
          countsByReason: {
            not_in_pack: this._categorizedTotals.byReason.not_in_pack,
            reflection_failed: this._categorizedTotals.byReason.reflection_failed,
            compile_failed: this._categorizedTotals.byReason.compile_failed,
            other: this._categorizedTotals.byReason.other,
          },
          countsByStage: {
            vertex: this._categorizedTotals.byStage.vertex,
            fragment: this._categorizedTotals.byStage.fragment,
            compute: this._categorizedTotals.byStage.compute,
          },
          uniqueCountsByReason: uniqueCounts.byReason,
          uniqueCountsByStage: uniqueCounts.byStage,
          realShaderGuard: {
            registeredRealShaders: this._realShaders.size,
            blockedOverrideCount: this._blockedOverrideCount,
            overflowBlockedCount: this._blockedOverflowCount,
          },
          topMissing,
        };
        if (this._summaryCache.size >= 16) {
          const oldestKey = this._summaryCache.keys().next().value;
          this._summaryCache.delete(oldestKey);
        }
        this._summaryCache.set(cacheKey, cached);
      }

      const cachedTop = cached.topMissing;
      const clonedTop = new Array(cachedTop.length);
      for (let i = 0; i < cachedTop.length; i += 1) {
        const row = cachedTop[i];
        clonedTop[i] = {
          hashHex: row.hashHex,
          stage: row.stage,
          reason: row.reason,
          lastReason: row.lastReason,
          hitCount: row.hitCount,
          reasons: {
            not_in_pack: row.reasons.not_in_pack,
            reflection_failed: row.reasons.reflection_failed,
            compile_failed: row.reasons.compile_failed,
            other: row.reasons.other,
          },
          firstSeenMs: row.firstSeenMs,
          lastSeenMs: row.lastSeenMs,
        };
      }

      return {
        totalMissingEvents: cached.totalMissingEvents,
        uniqueMissingHashes: cached.uniqueMissingHashes,
        maxTrackedHashes: cached.maxTrackedHashes,
        overflowCount: cached.overflowCount,
        countsByReason: {
          not_in_pack: cached.countsByReason.not_in_pack,
          reflection_failed: cached.countsByReason.reflection_failed,
          compile_failed: cached.countsByReason.compile_failed,
          other: cached.countsByReason.other,
        },
        countsByStage: {
          vertex: cached.countsByStage.vertex,
          fragment: cached.countsByStage.fragment,
          compute: cached.countsByStage.compute,
        },
        uniqueCountsByReason: {
          not_in_pack: cached.uniqueCountsByReason.not_in_pack,
          reflection_failed: cached.uniqueCountsByReason.reflection_failed,
          compile_failed: cached.uniqueCountsByReason.compile_failed,
          other: cached.uniqueCountsByReason.other,
        },
        uniqueCountsByStage: {
          vertex: cached.uniqueCountsByStage.vertex,
          fragment: cached.uniqueCountsByStage.fragment,
          compute: cached.uniqueCountsByStage.compute,
        },
        realShaderGuard: {
          registeredRealShaders: cached.realShaderGuard.registeredRealShaders,
          blockedOverrideCount: cached.realShaderGuard.blockedOverrideCount,
          overflowBlockedCount: cached.realShaderGuard.overflowBlockedCount,
        },
        topMissing: clonedTop,
      };
    }

    /**
     * Return top missing shaders sorted by hit count descending (JSON-serializable).
     *
     * @param {{ limit?: number }} [options]
     * @returns {Array<{ hash: string, stage: string, hitCount: number, firstSeenMs: number, lastSeenMs: number }>}
     */
    getMissingReport(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      let limit = opts.limit;
      if (limit === undefined || limit === null) {
        limit = this._missingCount;
      }
      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }
      if (limit === 0 || this._missingCount === 0) {
        return [];
      }

      const selected = selectTopMissingRows(this._missingByStage, this._missingCount, limit);
      const out = new Array(selected.length);
      for (let i = 0; i < selected.length; i += 1) {
        const row = selected[i];
        out[i] = {
          hash: row.hash,
          stage: row.stage,
          hitCount: row.hitCount,
          firstSeenMs: row.firstSeenMs,
          lastSeenMs: row.lastSeenMs,
        };
      }
      return out;
    }

    /**
     * Return top-N missing shaders ranked by miss count descending, breaking ties
     * deterministically by `hash` / `hashHex` ascending (then `stage` ascending).
     * Records are strictly sanitized (`{ hash, hashHex, stage, count, firstSeenFrame, lastSeenFrame }`)
     * with zero host paths, raw shader text, or binary dumps.
     *
     * @param {number} [limit=16]
     * @returns {Array<{
     *   hash: string,
     *   hashHex: string,
     *   stage: 'vertex' | 'fragment' | 'compute',
     *   count: number,
     *   firstSeenFrame: number,
     *   lastSeenFrame: number,
     * }>}
     */
    getTopMissingShaders(limit = DEFAULT_TOP_MISSING_LIMIT) {
      const resolvedLimit =
        limit === undefined || limit === null ? DEFAULT_TOP_MISSING_LIMIT : limit;
      if (!Number.isInteger(resolvedLimit) || resolvedLimit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${resolvedLimit}`);
      }
      if (resolvedLimit === 0 || this._missingCount === 0) {
        return [];
      }

      const selected = selectTopMissingRows(
        this._missingByStage,
        this._missingCount,
        resolvedLimit
      );
      const out = new Array(selected.length);
      for (let i = 0; i < selected.length; i += 1) {
        const row = selected[i];
        out[i] = {
          hash: row.hash,
          hashHex: row.hash,
          stage: row.stage,
          count: row.hitCount,
          firstSeenFrame: row.firstSeenFrame,
          lastSeenFrame: row.lastSeenFrame,
        };
      }
      return out;
    }

    /**
     * Return the bounded sanitized frequency list of missing/fallback shader hashes
     * sorted by descending `hitCount`, breaking ties deterministically by ascending `hash`
     * (`0x<16-hex>`). Ready for Main Menu / In-Game shader completeness triage and S47 export.
     *
     * @param {number | { limit?: number | null } | null} [optionsOrLimit]
     * @returns {Array<{
     *   hash: string,
     *   hashHex: string,
     *   hitCount: number,
     *   stage: 'vertex' | 'fragment' | 'compute' | null,
     *   stages: Array<'vertex' | 'fragment' | 'compute'>,
     *   firstSeenMs: number,
     *   lastSeenMs: number,
     * }>}
     */
    getMissingShaderPriorityList(optionsOrLimit) {
      let limit = this._missingPriorityHashes.size;
      if (typeof optionsOrLimit === 'number') {
        limit = optionsOrLimit;
      } else if (optionsOrLimit !== undefined && optionsOrLimit !== null) {
        if (typeof optionsOrLimit !== 'object') {
          throw new RangeError(
            `options must be an object or non-negative integer limit, got ${optionsOrLimit}`
          );
        }
        if (optionsOrLimit.limit !== undefined && optionsOrLimit.limit !== null) {
          limit = optionsOrLimit.limit;
        }
      }
      if (!Number.isInteger(limit) || limit < 0) {
        throw new RangeError(`limit must be a non-negative integer, got ${limit}`);
      }
      if (limit === 0 || this._missingPriorityHashes.size === 0) {
        return [];
      }

      const rows = Array.from(this._missingPriorityHashes.values());
      rows.sort((a, b) => {
        if (a.hitCount !== b.hitCount) {
          return b.hitCount - a.hitCount;
        }
        return a.hashHex < b.hashHex ? -1 : a.hashHex > b.hashHex ? 1 : 0;
      });

      const count = limit < rows.length ? limit : rows.length;
      const out = new Array(count);
      for (let i = 0; i < count; i += 1) {
        const row = rows[i];
        const stages = [];
        if (row.stageMask & (1 << SHADER_STAGE.VERTEX)) stages.push('vertex');
        if (row.stageMask & (1 << SHADER_STAGE.FRAGMENT)) stages.push('fragment');
        if (row.stageMask & (1 << SHADER_STAGE.COMPUTE)) stages.push('compute');
        out[i] = {
          hash: row.hashHex,
          hashHex: row.hashHex,
          hitCount: row.hitCount,
          stage: row.lastStage,
          stages,
          firstSeenMs: row.firstSeenMs,
          lastSeenMs: row.lastSeenMs,
        };
      }
      return out;
    }

    /**
     * Format the missing/fallback shader priority table as `# header` + `0x<16-hex>` lines
     * directly consumable by `playgta6-shaders pack-rdna2 --priority-hashes` (S47).
     * Strictly sanitized: contains only `#` header metadata and canonical 64-bit hex hashes.
     *
     * @param {number | { limit?: number | null, includeHitComments?: boolean } | null} [optionsOrLimit]
     * @returns {string}
     */
    exportPriorityHashesText(optionsOrLimit) {
      let includeHitComments = false;
      if (
        optionsOrLimit !== undefined &&
        optionsOrLimit !== null &&
        typeof optionsOrLimit === 'object'
      ) {
        includeHitComments = optionsOrLimit.includeHitComments === true;
      }

      const list = this.getMissingShaderPriorityList(optionsOrLimit);
      let totalHits = 0;
      for (let i = 0; i < list.length; i += 1) {
        totalHits += list[i].hitCount;
      }

      const lines = [
        '# playgta6-shaders pack-rdna2 --priority-hashes (W304/S47)',
        `# unique_hashes=${list.length} total_hits=${totalHits}`,
      ];
      for (let i = 0; i < list.length; i += 1) {
        const entry = list[i];
        if (includeHitComments) {
          lines.push(`${entry.hash} # hits=${entry.hitCount}`);
        } else {
          lines.push(entry.hash);
        }
      }
      return `${lines.join('\n')}\n`;
    }

    /**
     * Aggregate lookup / missing-shader telemetry.
     */
    getStats() {
      const uniqueMissing = this._missingCount;
      const total = this._stats.totalLookups;
      const hitRatio = total > 0 ? this._stats.packHits / total : 0;
      const hitRatePct =
        total > 0 ? Math.round(hitRatio * 10000) / 100 : 0;
      const stats = {
        totalLookups: this._stats.totalLookups,
        packHits: this._stats.packHits,
        fallbackHits: this._stats.fallbackHits,
        builtinHits: this._stats.builtinHits,
        uniqueMissing,
        missingUniqueHashes: this._missingPriorityHashes.size,
        fastPathHits: this._fastPathHits,
        overflowMissingCount: this._overflowMissingCount,
        overflowMissingHashes: this._overflowMissingCount,
        hitRatePct,
        hitRatio,
        compileFailureEntries: this._compileFailures.size,
        compileFailureCacheHits: this._compileFailureCacheHits,
        compileFailureRecorded: this._compileFailureRecorded,
      };
      Object.defineProperty(stats, 'rangedPendingHits', {
        value: this._rangedPendingHits,
        enumerable: false,
        configurable: true,
      });
      return stats;
    }

    /**
     * Clear tracked missing-shader maps, categorized missing telemetry, priority hash table,
     * and overflow counters.
     * Preserves configured `packLoader`, `fallbackHashes`, registered real shaders,
     * lookup hit counters, and memoized shader resolutions.
     */
    resetMissingTelemetry() {
      this._missingByStage[1].clear();
      this._missingByStage[2].clear();
      this._missingByStage[3].clear();
      this._missingCount = 0;
      this._overflowMissingCount = 0;
      this._telemetryGeneration += 1;

      this._missingPriorityHashes.clear();
      this._missingPrioritySeq = 0;
      this._missingPriorityEvictions = 0;

      this._categorizedMissing.clear();
      this._categorizedOverflowCount = 0;
      this._categorizedTotals.total = 0;
      this._categorizedTotals.byReason = createEmptyReasonCounts();
      this._categorizedTotals.byStage = createEmptyStageCounts();
      this._summaryVersion += 1;
    }

    /**
     * Clear missing-shader telemetry, categorized summary, priority hash table, guard counters,
     * negative compile-failure cache, and lookup counters.
     * Preserves configured `packLoader`, `fallbackHashes`, and registered real shaders.
     */
    reset() {
      this.resetMissingTelemetry();
      this._stats.totalLookups = 0;
      this._stats.packHits = 0;
      this._stats.fallbackHits = 0;
      this._stats.builtinHits = 0;
      this._rangedPendingHits = 0;

      this._resolveCacheHits = 0;
      this._fastPathHits = 0;
      this._resolveCacheMisses = 0;

      this._blockedOverrides.clear();
      this._blockedOverrideCount = 0;
      this._blockedOverflowCount = 0;
      this._summaryVersion += 1;

      if (this._compileFailures.size > 0) {
        this._compileFailures.clear();
        this._clearResolveCache();
      }
      this._compileFailureCacheHits = 0;
      this._compileFailureRecorded = 0;
    }

    /**
     * Alias for `reset()` that resets lookup counters, missing-shader telemetry,
     * priority hash table, and compile-failure cache statistics.
     */
    resetStats() {
      this.reset();
    }
  }

  ShaderFallbackRouter.getBuiltInFallbackProfiles = getBuiltInFallbackProfiles;
  ShaderFallbackRouter.buildInterstageFallbackVertexWgsl = buildInterstageFallbackVertexWgsl;

  const exportedApi = Object.freeze({
    SHADER_STAGE,
    STAGE_NAMES,
    MISSING_SHADER_REASONS,
    BUILTIN_FALLBACK_WGSL,
    BUILTIN_FALLBACK_PROFILES,
    DEFAULT_ENTRY_POINTS,
    DEFAULT_MAX_TRACKED_MISSING,
    DEFAULT_MAX_MISSING_HASHES,
    DEFAULT_MAX_TRACKED_MISSING_HASHES,
    DEFAULT_MAX_CATEGORIZED_MISSING_HASHES,
    DEFAULT_MAX_RESOLVE_CACHE_ENTRIES,
    DEFAULT_MAX_COMPILE_FAILURE_ENTRIES,
    DEFAULT_TOP_MISSING_LIMIT,
    DEFAULT_MISSING_SUMMARY_TOP_LIMIT,
    normalizeShaderHash,
    formatShaderHash,
    normalizeStage,
    normalizeMissingReason,
    sanitizeCompileFailureReason,
    normalizeFallbackProfile,
    selectDeterministicFallbackProfile,
    getBuiltInFallbackProfiles,
    buildInterstageFallbackVertexWgsl,
    ShaderFallbackRouter,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ShaderFallbackRouter = exportedApi;
  } else if (root) {
    root.PlayGta6ShaderFallbackRouter = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
