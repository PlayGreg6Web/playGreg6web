'use strict';

// Browser-side async pipeline prewarm queue orchestrator & PG6W warm-list codec/store (#1613, #2111).
//
// Bridges the C++ runtime exports in `runtime/wasm_boot.cpp` & `runtime/gpu/pipeline_prewarm.{h,cpp}`:
//   - playgta6_prewarm_load_list(dataPtr, size) -> i32 queued count (or -1 if malformed)
//   - playgta6_prewarm_take(outPtr, max)        -> i32 count of 24-byte PrewarmTask structs
//   - playgta6_prewarm_complete(id, ok, ns)     -> i32 (1 if accepted, 0 if unknown/not in-flight)
//   - playgta6_prewarm_progress(outPtr)         -> void (fills 48-byte PrewarmProgress struct)
//   - playgta6_prewarm_save_list(outPtr, cap)   -> u64 byte size of PG6W warm list
//
// Compatible with classic Web Workers (importScripts -> globalThis.PlayGta6PrewarmBridge)
// and Node.js CommonJS/ESM tests (module.exports).

(function initPlayGta6PrewarmBridge(root) {
  const PREWARM_MAGIC = 'PG6W';
  const PREWARM_MAGIC_BYTES = Object.freeze([0x50, 0x47, 0x36, 0x57]);
  const PREWARM_LIST_VERSION_MODULES = 1;
  const PREWARM_LIST_VERSION_PIPELINES = 2;
  const PREWARM_LIST_HEADER_BYTES = 16;
  const PREWARM_LIST_ENTRY_BYTES = 24;
  const PREWARM_TASK_BYTES = 24;
  const PREWARM_PROGRESS_BYTES = 48;
  const PREWARM_KIND_GRAPHICS = 0;
  const PREWARM_KIND_COMPUTE = 1;
  const PREWARM_DEFAULT_MAX_IN_FLIGHT = 4;
  const PREWARM_HARD_MAX_IN_FLIGHT = 4;
  const PREWARM_MAX_ENTRIES = 1 << 20;
  const PREWARM_MAX_DETAIL_BYTES = 1 << 20;
  const PREWARM_DEFAULT_POLL_INTERVAL_MS = 100;
  const PREWARM_DEFAULT_SAVE_INTERVAL_MS = 15000;
  const PREWARM_DEFAULT_STORE_KEY = 'default';
  const PREWARM_DEFAULT_DB_NAME = 'playgta6-prewarm';
  const PREWARM_DEFAULT_STORE_NAME = 'warm_lists';
  const PREWARM_DEFAULT_OPFS_DIR = 'playgta6-prewarm';

  const PREWARM_MSG_REQUEST = 'prewarm-task';
  const PREWARM_MSG_RESULT = 'prewarm-task-result';
  const PREWARM_MSG_PROGRESS = 'prewarm-progress';
  const PREWARM_MSG_SAVE = 'prewarm-save-list';

  const U64_MASK = 0xffffffffffffffffn;
  const NS_PER_MS = 1e6;

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function toUint8View(input) {
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
    return null;
  }

  function normalizeU64(value, label = 'key') {
    if (typeof value === 'bigint') {
      if (value < 0n || value > U64_MASK) {
        throw new RangeError(`${label} out of uint64 range`);
      }
      return value;
    }
    if (typeof value === 'number') {
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new RangeError(`${label} must be a non-negative safe integer`);
      }
      return BigInt(value);
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (/^0x[0-9a-f]{1,16}$/i.test(trimmed)) {
        return BigInt(trimmed) & U64_MASK;
      }
      if (/^[0-9a-f]{16}$/i.test(trimmed)) {
        return BigInt(`0x${trimmed}`) & U64_MASK;
      }
      if (/^\d+$/.test(trimmed)) {
        const parsed = BigInt(trimmed);
        if (parsed >= 0n && parsed <= U64_MASK) {
          return parsed;
        }
      }
    }
    throw new TypeError(`${label} must be a BigInt, non-negative safe integer, or uint64 string`);
  }

  function formatShaderKeyHex(key) {
    try {
      const bi = normalizeU64(key, 'shaderKey');
      return `0x${bi.toString(16).padStart(16, '0')}`;
    } catch {
      return '0x0000000000000000';
    }
  }

  function normalizeKind(kind) {
    if (kind === PREWARM_KIND_GRAPHICS || kind === 'graphics' || kind === 'render') {
      return PREWARM_KIND_GRAPHICS;
    }
    if (kind === PREWARM_KIND_COMPUTE || kind === 'compute') {
      return PREWARM_KIND_COMPUTE;
    }
    return -1;
  }

  function kindToString(kind) {
    return kind === PREWARM_KIND_COMPUTE ? 'compute' : 'graphics';
  }

  function safeBigIntToNumber(value) {
    if (typeof value === 'bigint') {
      return Number(value);
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    return 0;
  }

  /**
   * Parse and validate a binary `'PG6W'` warm list (v1 module-only or v2 full pipelines).
   * Mirrors `PipelinePrewarm::load_warm_list` in `runtime/gpu/pipeline_prewarm.cpp`.
   *
   * @param {Uint8Array | ArrayBuffer | ArrayBufferView} input
   * @returns {{ ok: true, version: number, count: number, byteLength: number, entries: Array<{ kind: number, key0: bigint, key1: bigint, detail: Uint8Array | null }> } | { ok: false, error: string }}
   */
  function parseWarmList(input) {
    const bytes = toUint8View(input);
    if (!bytes || bytes.byteLength < PREWARM_LIST_HEADER_BYTES) {
      return { ok: false, error: 'warm list too short' };
    }
    if (
      bytes[0] !== PREWARM_MAGIC_BYTES[0] ||
      bytes[1] !== PREWARM_MAGIC_BYTES[1] ||
      bytes[2] !== PREWARM_MAGIC_BYTES[2] ||
      bytes[3] !== PREWARM_MAGIC_BYTES[3]
    ) {
      return { ok: false, error: 'warm list bad magic' };
    }

    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = dv.getUint32(4, true);
    if (version !== PREWARM_LIST_VERSION_MODULES && version !== PREWARM_LIST_VERSION_PIPELINES) {
      return { ok: false, error: 'warm list unsupported version' };
    }

    const count = dv.getUint32(8, true);
    if (count > PREWARM_MAX_ENTRIES) {
      return { ok: false, error: 'warm list too many entries' };
    }

    const minSize = PREWARM_LIST_HEADER_BYTES + count * PREWARM_LIST_ENTRY_BYTES;
    if (bytes.byteLength < minSize) {
      return { ok: false, error: 'warm list truncated' };
    }

    const entries = new Array(count);
    let pos = PREWARM_LIST_HEADER_BYTES;
    for (let i = 0; i < count; i += 1) {
      if (bytes.byteLength - pos < PREWARM_LIST_ENTRY_BYTES) {
        return { ok: false, error: 'warm list truncated' };
      }
      const kind = bytes[pos];
      const detailLen = version === PREWARM_LIST_VERSION_PIPELINES ? dv.getUint32(pos + 4, true) : 0;
      const key0 = dv.getBigUint64(pos + 8, true);
      const key1 = dv.getBigUint64(pos + 16, true);
      pos += PREWARM_LIST_ENTRY_BYTES;

      if (kind > PREWARM_KIND_COMPUTE || key0 === 0n) {
        return { ok: false, error: 'warm list bad entry' };
      }

      let detail = null;
      if (version === PREWARM_LIST_VERSION_PIPELINES) {
        if (detailLen > PREWARM_MAX_DETAIL_BYTES) {
          return { ok: false, error: 'warm list bad entry' };
        }
        if (bytes.byteLength - pos < detailLen) {
          return { ok: false, error: 'warm list truncated' };
        }
        if (detailLen > 0) {
          detail = bytes.slice(pos, pos + detailLen);
        }
        pos += detailLen;
      }

      entries[i] = { kind, key0, key1, detail };
    }

    return {
      ok: true,
      version,
      count,
      byteLength: pos,
      entries,
    };
  }

  /**
   * Serialize warm-list entries in first-use order into a `'PG6W'` binary buffer.
   * Uses v1 (24-byte entries) when no entry carries a `detail` payload, and v2 otherwise.
   *
   * @param {Array<{ kind: number | string, key0: bigint | number | string, key1?: bigint | number | string, detail?: Uint8Array | ArrayBuffer | null }>} entries
   * @param {{ version?: 1 | 2 }} [options]
   * @returns {Uint8Array}
   */
  function serializeWarmList(entries, options = {}) {
    if (!Array.isArray(entries)) {
      throw new TypeError('entries must be an array');
    }
    if (entries.length > PREWARM_MAX_ENTRIES) {
      throw new RangeError(`warm list entry count exceeds limit (${PREWARM_MAX_ENTRIES})`);
    }

    const normalized = new Array(entries.length);
    let anyDetail = options && options.version === PREWARM_LIST_VERSION_PIPELINES;
    let totalDetailBytes = 0;

    for (let i = 0; i < entries.length; i += 1) {
      const raw = entries[i];
      if (!raw || typeof raw !== 'object') {
        throw new TypeError(`warm list entry ${i} must be an object`);
      }
      const kind = normalizeKind(raw.kind);
      if (kind < 0) {
        throw new RangeError(`warm list entry ${i} has invalid kind: ${raw.kind}`);
      }
      const key0 = normalizeU64(raw.key0, `entry[${i}].key0`);
      if (key0 === 0n) {
        throw new RangeError(`warm list entry ${i} key0 must be non-zero`);
      }
      const key1 =
        raw.key1 === undefined || raw.key1 === null
          ? 0n
          : normalizeU64(raw.key1, `entry[${i}].key1`);

      let detail = null;
      if (raw.detail !== undefined && raw.detail !== null) {
        const view = toUint8View(raw.detail);
        if (!view) {
          throw new TypeError(`warm list entry ${i} detail must be binary`);
        }
        if (view.byteLength > PREWARM_MAX_DETAIL_BYTES) {
          throw new RangeError(`warm list entry ${i} detail exceeds ${PREWARM_MAX_DETAIL_BYTES} bytes`);
        }
        if (view.byteLength > 0) {
          detail = view;
          anyDetail = true;
          totalDetailBytes += view.byteLength;
        }
      }

      normalized[i] = { kind, key0, key1, detail };
    }

    const version = anyDetail ? PREWARM_LIST_VERSION_PIPELINES : PREWARM_LIST_VERSION_MODULES;
    const totalBytes =
      PREWARM_LIST_HEADER_BYTES +
      normalized.length * PREWARM_LIST_ENTRY_BYTES +
      (anyDetail ? totalDetailBytes : 0);

    const out = new Uint8Array(totalBytes);
    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);

    out[0] = PREWARM_MAGIC_BYTES[0];
    out[1] = PREWARM_MAGIC_BYTES[1];
    out[2] = PREWARM_MAGIC_BYTES[2];
    out[3] = PREWARM_MAGIC_BYTES[3];
    dv.setUint32(4, version, true);
    dv.setUint32(8, normalized.length, true);
    dv.setUint32(12, 0, true);

    let pos = PREWARM_LIST_HEADER_BYTES;
    for (let i = 0; i < normalized.length; i += 1) {
      const e = normalized[i];
      out[pos] = e.kind;
      if (anyDetail) {
        dv.setUint32(pos + 4, e.detail ? e.detail.byteLength : 0, true);
      }
      dv.setBigUint64(pos + 8, e.key0, true);
      dv.setBigUint64(pos + 16, e.key1, true);
      pos += PREWARM_LIST_ENTRY_BYTES;
      if (anyDetail && e.detail) {
        out.set(e.detail, pos);
        pos += e.detail.byteLength;
      }
    }

    return out;
  }

  /**
   * Decode `count` 24-byte `PrewarmTask` records from linear memory (`{u32 id, u32 kind, u64 key0, u64 key1}`).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} mem
   * @param {number} byteOffset
   * @param {number} count
   * @returns {Array<{ id: number, kind: number, key0: bigint, key1: bigint }>}
   */
  function decodePrewarmTasks(mem, byteOffset, count) {
    const view = toUint8View(mem);
    if (!view || !Number.isInteger(count) || count <= 0) {
      return [];
    }
    const base = Number(byteOffset);
    const needed = count * PREWARM_TASK_BYTES;
    if (!Number.isSafeInteger(base) || base < 0 || base + needed > view.byteLength) {
      throw new RangeError('PrewarmTask buffer out of bounds');
    }
    const dv = new DataView(view.buffer, view.byteOffset + base, needed);
    const tasks = new Array(count);
    for (let i = 0; i < count; i += 1) {
      const off = i * PREWARM_TASK_BYTES;
      tasks[i] = {
        id: dv.getUint32(off, true),
        kind: dv.getUint32(off + 4, true),
        key0: dv.getBigUint64(off + 8, true),
        key1: dv.getBigUint64(off + 16, true),
      };
    }
    return tasks;
  }

  /**
   * Decode a 48-byte `PrewarmProgress` struct from linear memory:
   * `{total, queued, in_flight, done, failed, skipped, observed, reserved, u64 longest_ns, u64 total_ns}`.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} mem
   * @param {number} byteOffset
   * @returns {{
   *   total: number,
   *   queued: number,
   *   in_flight: number,
   *   done: number,
   *   failed: number,
   *   skipped: number,
   *   observed: number,
   *   reserved: number,
   *   longest_ns: number,
   *   total_ns: number
   * }}
   */
  function decodePrewarmProgress(mem, byteOffset = 0) {
    const view = toUint8View(mem);
    const base = Number(byteOffset);
    if (
      !view ||
      !Number.isSafeInteger(base) ||
      base < 0 ||
      base + PREWARM_PROGRESS_BYTES > view.byteLength
    ) {
      throw new RangeError('PrewarmProgress buffer out of bounds');
    }
    const dv = new DataView(view.buffer, view.byteOffset + base, PREWARM_PROGRESS_BYTES);
    return {
      total: dv.getUint32(0, true),
      queued: dv.getUint32(4, true),
      in_flight: dv.getUint32(8, true),
      done: dv.getUint32(12, true),
      failed: dv.getUint32(16, true),
      skipped: dv.getUint32(20, true),
      observed: dv.getUint32(24, true),
      reserved: dv.getUint32(28, true),
      longest_ns: safeBigIntToNumber(dv.getBigUint64(32, true)),
      total_ns: safeBigIntToNumber(dv.getBigUint64(40, true)),
    };
  }

  /**
   * Resolve a named C export from an Emscripten module or raw Wasm exports object.
   */
  function resolveExportFn(mod, baseName) {
    if (!mod || typeof mod !== 'object') {
      return null;
    }
    const prefixed = `_${baseName}`;
    if (typeof mod[prefixed] === 'function') {
      return mod[prefixed].bind(mod);
    }
    if (typeof mod[baseName] === 'function') {
      return mod[baseName].bind(mod);
    }
    if (mod.wasmExports && typeof mod.wasmExports === 'object') {
      if (typeof mod.wasmExports[baseName] === 'function') {
        return mod.wasmExports[baseName].bind(mod.wasmExports);
      }
      if (typeof mod.wasmExports[prefixed] === 'function') {
        return mod.wasmExports[prefixed].bind(mod.wasmExports);
      }
    }
    if (mod.asm && typeof mod.asm === 'object') {
      if (typeof mod.asm[baseName] === 'function') {
        return mod.asm[baseName].bind(mod.asm);
      }
      if (typeof mod.asm[prefixed] === 'function') {
        return mod.asm[prefixed].bind(mod.asm);
      }
    }
    return null;
  }

  function resolveMemoryBuffer(mod, getMemory) {
    if (typeof getMemory === 'function') {
      try {
        const mem = getMemory();
        if (mem) {
          if (mem.buffer && !mem.buffer.detached && mem.buffer.byteLength > 0) {
            return mem.buffer;
          }
          if (
            (mem instanceof ArrayBuffer ||
              (typeof SharedArrayBuffer !== 'undefined' && mem instanceof SharedArrayBuffer)) &&
            !mem.detached &&
            mem.byteLength > 0
          ) {
            return mem;
          }
        }
      } catch {
        // Fall through to module inspection.
      }
    }
    if (!mod || typeof mod !== 'object') {
      return null;
    }
    const candidates = [
      mod.wasmMemory,
      mod.memory,
      mod.HEAPU8 && mod.HEAPU8.buffer,
      mod.HEAP32 && mod.HEAP32.buffer,
      mod.wasmExports && mod.wasmExports.memory,
      mod.asm && mod.asm.memory,
    ];
    for (let i = 0; i < candidates.length; i += 1) {
      const c = candidates[i];
      if (!c) continue;
      const buf = c.buffer || c;
      if (
        buf &&
        !buf.detached &&
        typeof buf.byteLength === 'number' &&
        buf.byteLength > 0
      ) {
        return buf;
      }
    }
    return null;
  }

  /**
   * Call a Wasm export that accepts `uint64_t` arguments, preferring `BigInt` (Wasm64 / JS BigInt
   * integration) and falling back to `Number` if a lightweight JS test double rejects `BigInt`.
   */
  function callWithU64Fallback(fn, bigintArgs, numberArgs) {
    try {
      return fn(...bigintArgs);
    } catch (err) {
      if (err instanceof TypeError) {
        return fn(...numberArgs);
      }
      throw err;
    }
  }

  function resolvePrewarmWasmApi(mod, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const loadListFn = resolveExportFn(mod, 'playgta6_prewarm_load_list');
    const takeFn = resolveExportFn(mod, 'playgta6_prewarm_take');
    const completeFn = resolveExportFn(mod, 'playgta6_prewarm_complete');
    const progressFn = resolveExportFn(mod, 'playgta6_prewarm_progress');
    const saveListFn = resolveExportFn(mod, 'playgta6_prewarm_save_list');

    const mallocFn =
      typeof opts.alloc === 'function' ? opts.alloc : resolveExportFn(mod, 'malloc');
    const freeFn = typeof opts.free === 'function' ? opts.free : resolveExportFn(mod, 'free');
    const stackSaveFn = resolveExportFn(mod, 'stackSave');
    const stackAllocFn = resolveExportFn(mod, 'stackAlloc');
    const stackRestoreFn = resolveExportFn(mod, 'stackRestore');
    const scratchOffset =
      Number.isInteger(opts.scratchOffset) && opts.scratchOffset >= 0 ? opts.scratchOffset : null;
    const scratchSize =
      Number.isInteger(opts.scratchSize) && opts.scratchSize > 0 ? opts.scratchSize : 0;

    const explicitWasm64 =
      typeof opts.wasm64 === 'boolean'
        ? opts.wasm64
        : opts.pointerSize === 8
          ? true
          : opts.pointerSize === 4
            ? false
            : null;
    let detectedBigIntPtr = explicitWasm64;

    function toPtrArg(ptr) {
      if (detectedBigIntPtr === true) {
        return typeof ptr === 'bigint' ? ptr : BigInt(ptr);
      }
      if (detectedBigIntPtr === false) {
        return typeof ptr === 'bigint' ? Number(ptr) : ptr;
      }
      return ptr;
    }

    function ptrToOffset(ptr) {
      if (typeof ptr === 'bigint') {
        if (detectedBigIntPtr === null) {
          detectedBigIntPtr = true;
        }
        return Number(ptr);
      }
      if (typeof ptr === 'number' && Number.isFinite(ptr)) {
        if (detectedBigIntPtr === null && ptr > 0) {
          detectedBigIntPtr = false;
        }
        return ptr;
      }
      return 0;
    }

    function withScratch(byteLength, fn) {
      const size = Math.max(8, Math.ceil(byteLength / 8) * 8);

      if (mallocFn && freeFn) {
        const rawPtr = callWithU64Fallback(
          mallocFn,
          detectedBigIntPtr === false ? [size] : [BigInt(size)],
          [size],
        );
        const offset = ptrToOffset(rawPtr);
        if (!Number.isSafeInteger(offset) || offset <= 0) {
          throw new Error(`prewarm scratch malloc(${size}) failed`);
        }
        try {
          return fn(rawPtr, offset);
        } finally {
          try {
            freeFn(rawPtr);
          } catch {
            // Ignore secondary free error.
          }
        }
      }

      if (stackSaveFn && stackAllocFn && stackRestoreFn) {
        const sp = stackSaveFn();
        try {
          const rawPtr = callWithU64Fallback(
            stackAllocFn,
            detectedBigIntPtr === false ? [size] : [BigInt(size)],
            [size],
          );
          const offset = ptrToOffset(rawPtr);
          if (!Number.isSafeInteger(offset) || offset <= 0) {
            throw new Error(`prewarm scratch stackAlloc(${size}) failed`);
          }
          return fn(rawPtr, offset);
        } finally {
          try {
            stackRestoreFn(sp);
          } catch {
            // Ignore secondary stackRestore error.
          }
        }
      }

      if (scratchOffset !== null && (scratchSize === 0 || size <= scratchSize)) {
        const rawPtr = detectedBigIntPtr === true ? BigInt(scratchOffset) : scratchOffset;
        return fn(rawPtr, scratchOffset);
      }

      throw new Error('prewarm bridge requires malloc/free, stackAlloc, or scratchOffset');
    }

    const available = Boolean(
      loadListFn || takeFn || completeFn || progressFn || saveListFn,
    );

    return {
      available,
      hasLoadList: Boolean(loadListFn),
      hasTake: Boolean(takeFn),
      hasComplete: Boolean(completeFn),
      hasProgress: Boolean(progressFn),
      hasSaveList: Boolean(saveListFn),
      getMemoryBuffer: () => resolveMemoryBuffer(mod, opts.getMemory),
      loadList(bytes) {
        if (!loadListFn) {
          return -1;
        }
        const view = toUint8View(bytes);
        if (!view || view.byteLength === 0) {
          return -1;
        }
        return withScratch(view.byteLength, (ptr, offset) => {
          const buf = resolveMemoryBuffer(mod, opts.getMemory);
          if (!buf || offset + view.byteLength > buf.byteLength) {
            throw new RangeError('Wasm memory too small for warm list');
          }
          new Uint8Array(buf, offset, view.byteLength).set(view);
          const raw = callWithU64Fallback(
            loadListFn,
            [toPtrArg(ptr), BigInt(view.byteLength)],
            [offset, view.byteLength],
          );
          return Number(raw);
        });
      },
      take(maxCount) {
        if (!takeFn) {
          return [];
        }
        const max = Math.max(1, Math.min(PREWARM_HARD_MAX_IN_FLIGHT, Math.floor(maxCount)));
        const outBytes = max * PREWARM_TASK_BYTES;
        return withScratch(outBytes, (ptr, offset) => {
          const rawCount = callWithU64Fallback(
            takeFn,
            [toPtrArg(ptr), max],
            [offset, max],
          );
          const count = Number(rawCount);
          if (!Number.isInteger(count) || count <= 0) {
            return [];
          }
          const clamped = Math.min(count, max);
          const buf = resolveMemoryBuffer(mod, opts.getMemory);
          if (!buf) {
            return [];
          }
          return decodePrewarmTasks(buf, offset, clamped);
        });
      },
      complete(id, ok, buildNs) {
        if (!completeFn) {
          return false;
        }
        const taskId = Number(id) >>> 0;
        const okInt = ok ? 1 : 0;
        const safeNs =
          typeof buildNs === 'bigint'
            ? buildNs < 0n
              ? 0n
              : buildNs
            : BigInt(Math.max(0, Math.round(Number.isFinite(buildNs) ? buildNs : 0)));
        const raw = callWithU64Fallback(
          completeFn,
          [taskId, okInt, safeNs],
          [taskId, okInt, Number(safeNs)],
        );
        return Number(raw) !== 0;
      },
      progress() {
        if (!progressFn) {
          return null;
        }
        return withScratch(PREWARM_PROGRESS_BYTES, (ptr, offset) => {
          callWithU64Fallback(progressFn, [toPtrArg(ptr)], [offset]);
          const buf = resolveMemoryBuffer(mod, opts.getMemory);
          if (!buf) {
            return null;
          }
          return decodePrewarmProgress(buf, offset);
        });
      },
      saveList() {
        if (!saveListFn) {
          return null;
        }
        let required = 0;
        if (detectedBigIntPtr === false) {
          required = safeBigIntToNumber(
            callWithU64Fallback(saveListFn, [0, 0n], [0, 0]),
          );
        } else {
          try {
            required = safeBigIntToNumber(saveListFn(0n, 0n));
            if (detectedBigIntPtr === null) {
              detectedBigIntPtr = true;
            }
          } catch (err) {
            if (!(err instanceof TypeError)) {
              throw err;
            }
            required = safeBigIntToNumber(
              callWithU64Fallback(saveListFn, [0, 0n], [0, 0]),
            );
            if (detectedBigIntPtr === null) {
              detectedBigIntPtr = false;
            }
          }
        }
        if (!Number.isSafeInteger(required) || required <= 0) {
          return null;
        }

        for (let attempt = 0; attempt < 2; attempt += 1) {
          const cap = required;
          const result = withScratch(cap, (ptr, offset) => {
            const actual = safeBigIntToNumber(
              callWithU64Fallback(saveListFn, [toPtrArg(ptr), BigInt(cap)], [offset, cap]),
            );
            if (!Number.isSafeInteger(actual) || actual <= 0) {
              return { done: true, bytes: null, needed: 0 };
            }
            if (actual > cap) {
              return { done: false, bytes: null, needed: actual };
            }
            const buf = resolveMemoryBuffer(mod, opts.getMemory);
            if (!buf || offset + actual > buf.byteLength) {
              return { done: true, bytes: null, needed: 0 };
            }
            const copy = new Uint8Array(actual);
            copy.set(new Uint8Array(buf, offset, actual));
            return { done: true, bytes: copy, needed: actual };
          });
          if (result.done) {
            return result.bytes;
          }
          required = result.needed;
        }
        return null;
      },
    };
  }

  function sanitizeStoreKey(key) {
    const raw = typeof key === 'string' && key.trim().length > 0 ? key.trim() : PREWARM_DEFAULT_STORE_KEY;
    return raw.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96) || PREWARM_DEFAULT_STORE_KEY;
  }

  /**
   * Persistent storage for `'PG6W'` warm lists with automatic OPFS -> IndexedDB -> in-memory fallback.
   */
  class PrewarmListStore {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      this._mode =
        opts.mode === 'opfs' || opts.mode === 'indexeddb' || opts.mode === 'memory'
          ? opts.mode
          : 'auto';
      this._dbName =
        typeof opts.dbName === 'string' && opts.dbName.length > 0
          ? opts.dbName
          : PREWARM_DEFAULT_DB_NAME;
      this._storeName =
        typeof opts.storeName === 'string' && opts.storeName.length > 0
          ? opts.storeName
          : PREWARM_DEFAULT_STORE_NAME;
      this._opfsDirName =
        typeof opts.opfsDirName === 'string' && opts.opfsDirName.length > 0
          ? opts.opfsDirName
          : PREWARM_DEFAULT_OPFS_DIR;
      this._storageAdapter =
        opts.storage && typeof opts.storage === 'object' ? opts.storage : null;
      this._navigator = opts.navigator !== undefined ? opts.navigator : (typeof navigator !== 'undefined' ? navigator : null);
      this._indexedDB = opts.indexedDB !== undefined ? opts.indexedDB : (typeof indexedDB !== 'undefined' ? indexedDB : null);
      this._memory = new Map();
      this._lastBackend = 'none';
      this._corruptReads = 0;
    }

    get lastBackend() {
      return this._lastBackend;
    }

    get corruptReads() {
      return this._corruptReads;
    }

    async _tryOpfsLoad(safeKey) {
      if (
        !this._navigator ||
        !this._navigator.storage ||
        typeof this._navigator.storage.getDirectory !== 'function'
      ) {
        return { supported: false, bytes: null };
      }
      const rootDir = await this._navigator.storage.getDirectory();
      const dir = await rootDir.getDirectoryHandle(this._opfsDirName, { create: false });
      const fileHandle = await dir.getFileHandle(`${safeKey}.pg6w`, { create: false });
      const file = await fileHandle.getFile();
      const buf = await file.arrayBuffer();
      return { supported: true, bytes: new Uint8Array(buf) };
    }

    async _tryOpfsSave(safeKey, bytes) {
      if (
        !this._navigator ||
        !this._navigator.storage ||
        typeof this._navigator.storage.getDirectory !== 'function'
      ) {
        return false;
      }
      const rootDir = await this._navigator.storage.getDirectory();
      const dir = await rootDir.getDirectoryHandle(this._opfsDirName, { create: true });
      const fileHandle = await dir.getFileHandle(`${safeKey}.pg6w`, { create: true });
      if (typeof fileHandle.createWritable === 'function') {
        const writable = await fileHandle.createWritable();
        await writable.write(bytes);
        await writable.close();
        return true;
      }
      if (typeof fileHandle.createSyncAccessHandle === 'function') {
        const handle = await fileHandle.createSyncAccessHandle();
        try {
          handle.truncate(0);
          handle.write(bytes, { at: 0 });
          handle.flush();
        } finally {
          handle.close();
        }
        return true;
      }
      return false;
    }

    async _tryOpfsClear(safeKey) {
      if (
        !this._navigator ||
        !this._navigator.storage ||
        typeof this._navigator.storage.getDirectory !== 'function'
      ) {
        return false;
      }
      const rootDir = await this._navigator.storage.getDirectory();
      const dir = await rootDir.getDirectoryHandle(this._opfsDirName, { create: false });
      if (typeof dir.removeEntry === 'function') {
        await dir.removeEntry(`${safeKey}.pg6w`);
        return true;
      }
      return false;
    }

    _openIdb() {
      const idb = this._indexedDB;
      if (!idb || typeof idb.open !== 'function') {
        return Promise.resolve(null);
      }
      return new Promise((resolve, reject) => {
        try {
          const req = idb.open(this._dbName, 1);
          req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(this._storeName)) {
              db.createObjectStore(this._storeName);
            }
          };
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => reject(req.error || new Error('IndexedDB open failed'));
        } catch (err) {
          reject(err);
        }
      });
    }

    async _tryIdbLoad(safeKey) {
      const db = await this._openIdb();
      if (!db) {
        return { supported: false, bytes: null };
      }
      try {
        const value = await new Promise((resolve, reject) => {
          const tx = db.transaction(this._storeName, 'readonly');
          const store = tx.objectStore(this._storeName);
          const req = store.get(safeKey);
          req.onsuccess = () => resolve(req.result ?? null);
          req.onerror = () => reject(req.error || new Error('IndexedDB get failed'));
        });
        return { supported: true, bytes: value ? toUint8View(value)?.slice() || null : null };
      } finally {
        try {
          db.close();
        } catch {
          // Ignore close errors.
        }
      }
    }

    async _tryIdbSave(safeKey, bytes) {
      const db = await this._openIdb();
      if (!db) {
        return false;
      }
      try {
        await new Promise((resolve, reject) => {
          const tx = db.transaction(this._storeName, 'readwrite');
          const store = tx.objectStore(this._storeName);
          store.put(bytes.slice(), safeKey);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error || new Error('IndexedDB put failed'));
          tx.onabort = () => reject(tx.error || new Error('IndexedDB put aborted'));
        });
        return true;
      } finally {
        try {
          db.close();
        } catch {
          // Ignore close errors.
        }
      }
    }

    async _tryIdbClear(safeKey) {
      const db = await this._openIdb();
      if (!db) {
        return false;
      }
      try {
        await new Promise((resolve, reject) => {
          const tx = db.transaction(this._storeName, 'readwrite');
          const store = tx.objectStore(this._storeName);
          store.delete(safeKey);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error || new Error('IndexedDB delete failed'));
        });
        return true;
      } finally {
        try {
          db.close();
        } catch {
          // Ignore close errors.
        }
      }
    }

    async load(key = PREWARM_DEFAULT_STORE_KEY) {
      const safeKey = sanitizeStoreKey(key);
      let rawBytes = null;

      if (this._storageAdapter && typeof this._storageAdapter.load === 'function') {
        try {
          const loaded = await this._storageAdapter.load(safeKey);
          rawBytes = loaded ? toUint8View(loaded)?.slice() || null : null;
          this._lastBackend = 'custom';
        } catch {
          rawBytes = null;
        }
      } else {
        if (this._mode === 'auto' || this._mode === 'opfs') {
          try {
            const res = await this._tryOpfsLoad(safeKey);
            if (res.supported) {
              this._lastBackend = 'opfs';
              rawBytes = res.bytes;
            }
          } catch {
            // OPFS missing file or unavailable -> fall through.
          }
        }
        if (rawBytes === null && (this._mode === 'auto' || this._mode === 'indexeddb')) {
          try {
            const res = await this._tryIdbLoad(safeKey);
            if (res.supported) {
              this._lastBackend = 'indexeddb';
              rawBytes = res.bytes;
            }
          } catch {
            // IndexedDB unavailable -> fall through.
          }
        }
        if (rawBytes === null && this._memory.has(safeKey)) {
          this._lastBackend = 'memory';
          rawBytes = this._memory.get(safeKey).slice();
        }
      }

      if (!rawBytes || rawBytes.byteLength === 0) {
        return null;
      }

      const parsed = parseWarmList(rawBytes);
      if (!parsed.ok) {
        this._corruptReads += 1;
        return null;
      }
      return rawBytes;
    }

    async save(key = PREWARM_DEFAULT_STORE_KEY, inputBytes) {
      const safeKey = sanitizeStoreKey(key);
      const view = toUint8View(inputBytes);
      if (!view || view.byteLength === 0) {
        return { ok: false, backend: 'none', byteLength: 0, error: 'empty warm list' };
      }
      const parsed = parseWarmList(view);
      if (!parsed.ok) {
        return { ok: false, backend: 'none', byteLength: 0, error: parsed.error };
      }
      const copy = view.slice();

      if (this._storageAdapter && typeof this._storageAdapter.save === 'function') {
        try {
          await this._storageAdapter.save(safeKey, copy);
          this._lastBackend = 'custom';
          return { ok: true, backend: 'custom', byteLength: copy.byteLength };
        } catch (err) {
          return {
            ok: false,
            backend: 'custom',
            byteLength: 0,
            error: err instanceof Error ? err.message : String(err),
          };
        }
      }

      if (this._mode === 'auto' || this._mode === 'opfs') {
        try {
          if (await this._tryOpfsSave(safeKey, copy)) {
            this._memory.set(safeKey, copy.slice());
            this._lastBackend = 'opfs';
            return { ok: true, backend: 'opfs', byteLength: copy.byteLength };
          }
        } catch {
          // Fall through to IndexedDB / memory.
        }
      }

      if (this._mode === 'auto' || this._mode === 'indexeddb') {
        try {
          if (await this._tryIdbSave(safeKey, copy)) {
            this._memory.set(safeKey, copy.slice());
            this._lastBackend = 'indexeddb';
            return { ok: true, backend: 'indexeddb', byteLength: copy.byteLength };
          }
        } catch {
          // Fall through to memory.
        }
      }

      this._memory.set(safeKey, copy);
      this._lastBackend = 'memory';
      return { ok: true, backend: 'memory', byteLength: copy.byteLength };
    }

    async clear(key = PREWARM_DEFAULT_STORE_KEY) {
      const safeKey = sanitizeStoreKey(key);
      this._memory.delete(safeKey);
      if (this._storageAdapter && typeof this._storageAdapter.clear === 'function') {
        try {
          await this._storageAdapter.clear(safeKey);
          return true;
        } catch {
          return false;
        }
      }
      let cleared = false;
      if (this._mode === 'auto' || this._mode === 'opfs') {
        try {
          cleared = (await this._tryOpfsClear(safeKey)) || cleared;
        } catch {
          // Ignore missing OPFS entry.
        }
      }
      if (this._mode === 'auto' || this._mode === 'indexeddb') {
        try {
          cleared = (await this._tryIdbClear(safeKey)) || cleared;
        } catch {
          // Ignore missing IDB entry.
        }
      }
      return cleared;
    }
  }

  /**
   * Look up a WGSL shader source or pre-compiled module from a shader pack / map / loader object.
   */
  async function resolveShaderFromPack(shaderPack, keyBigInt) {
    if (!shaderPack || keyBigInt === 0n) {
      return null;
    }
    const hexKey = formatShaderKeyHex(keyBigInt);
    const bareHex = hexKey.slice(2);

    if (typeof shaderPack.getShader === 'function') {
      const res = await shaderPack.getShader(keyBigInt);
      if (res) return res;
    }
    if (typeof shaderPack.getWgsl === 'function') {
      const res = await shaderPack.getWgsl(keyBigInt);
      if (res) return res;
    }
    if (typeof shaderPack.loadShader === 'function') {
      const res = await shaderPack.loadShader(keyBigInt);
      if (res) return res;
    }
    if (typeof shaderPack.get === 'function') {
      const res =
        shaderPack.get(keyBigInt) ||
        shaderPack.get(hexKey) ||
        shaderPack.get(bareHex) ||
        shaderPack.get(keyBigInt.toString());
      if (res) return res;
    }
    if (typeof shaderPack === 'object') {
      const direct = shaderPack[hexKey] || shaderPack[bareHex] || shaderPack[keyBigInt.toString()];
      if (direct) return direct;
    }
    return null;
  }

  async function compileShaderEntry(device, pipelineCache, shaderEntry, label) {
    if (!shaderEntry) {
      return null;
    }
    if (typeof shaderEntry === 'object' && shaderEntry.module) {
      return shaderEntry.module;
    }
    const wgslCode =
      typeof shaderEntry === 'string'
        ? shaderEntry
        : typeof shaderEntry.wgsl === 'string'
          ? shaderEntry.wgsl
          : typeof shaderEntry.code === 'string'
            ? shaderEntry.code
            : null;
    if (!wgslCode) {
      return null;
    }

    if (pipelineCache && typeof pipelineCache.getOrCreateShaderModule === 'function') {
      return pipelineCache.getOrCreateShaderModule(label, wgslCode);
    }
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new Error('no GPUDevice available to compile shader module');
    }

    const mod = device.createShaderModule({ label, code: wgslCode });
    if (mod && typeof mod.getCompilationInfo === 'function') {
      const info = await mod.getCompilationInfo();
      const messages = info && Array.isArray(info.messages) ? info.messages : [];
      for (let i = 0; i < messages.length; i += 1) {
        if (messages[i] && messages[i].type === 'error') {
          throw new Error(messages[i].message || `WGSL compilation error in ${label}`);
        }
      }
    }
    return mod;
  }

  /**
   * Execute a single `PrewarmTask` (`{ id, kind, key0, key1 }`) on the GPU worker / processor side,
   * measuring `buildNs` and returning `{ id, ok, buildNs, error? }`.
   */
  async function executePrewarmTask(processor, task, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const nowMs = typeof opts.now === 'function' ? opts.now : defaultNowMs;
    const startMs = nowMs();

    const taskId = Number(task && task.id) >>> 0;
    const kind = normalizeKind(task && task.kind);

    try {
      const key0 = normalizeU64(task && task.key0, 'task.key0');
      const key1 =
        task && task.key1 !== undefined && task.key1 !== null
          ? normalizeU64(task.key1, 'task.key1')
          : 0n;

      if (kind < 0 || key0 === 0n) {
        throw new RangeError('invalid prewarm task');
      }

      if (typeof opts.buildTask === 'function') {
        const res = await opts.buildTask({ id: taskId, kind, key0, key1 });
        const elapsedNs = Math.max(0, Math.round((nowMs() - startMs) * NS_PER_MS));
        if (typeof res === 'boolean') {
          return { id: taskId, ok: res, buildNs: elapsedNs };
        }
        if (res && typeof res === 'object') {
          return {
            id: taskId,
            ok: res.ok !== false,
            buildNs:
              typeof res.buildNs === 'number' && Number.isFinite(res.buildNs) && res.buildNs >= 0
                ? Math.round(res.buildNs)
                : elapsedNs,
            error: res.ok === false && res.error ? String(res.error) : undefined,
          };
        }
        return { id: taskId, ok: true, buildNs: elapsedNs };
      }

      const proc = processor && typeof processor === 'object' ? processor : {};
      const device = opts.device || proc.device || null;
      const shaderPack = opts.shaderPack || proc.shaderPack || proc.wgslPack || null;
      const pipelineCache = opts.pipelineCache || proc.pipelineCache || null;

      if (kind === PREWARM_KIND_COMPUTE) {
        const csEntry = await resolveShaderFromPack(shaderPack, key0);
        if (!csEntry) {
          throw new Error(`compute shader ${formatShaderKeyHex(key0)} not found in pack`);
        }
        const csModule = await compileShaderEntry(
          device,
          pipelineCache,
          csEntry,
          `prewarm-cs-${formatShaderKeyHex(key0)}`,
        );
        if (!csModule) {
          throw new Error(`failed to compile compute shader ${formatShaderKeyHex(key0)}`);
        }
        if (pipelineCache && typeof pipelineCache.prewarmComputePipeline === 'function') {
          await pipelineCache.prewarmComputePipeline({
            key: key0,
            module: csModule,
            entryPoint: csEntry.entryPoint || 'main',
          });
        } else if (
          opts.buildComputePipeline !== false &&
          device &&
          typeof device.createComputePipelineAsync === 'function'
        ) {
          await device.createComputePipelineAsync({
            label: `prewarm-cs-${formatShaderKeyHex(key0)}`,
            layout: 'auto',
            compute: {
              module: csModule,
              entryPoint: csEntry.entryPoint || 'main',
            },
          });
        }
      } else {
        const vsEntry = await resolveShaderFromPack(shaderPack, key0);
        if (!vsEntry) {
          throw new Error(`vertex shader ${formatShaderKeyHex(key0)} not found in pack`);
        }
        const vsModule = await compileShaderEntry(
          device,
          pipelineCache,
          vsEntry,
          `prewarm-vs-${formatShaderKeyHex(key0)}`,
        );
        if (!vsModule) {
          throw new Error(`failed to compile vertex shader ${formatShaderKeyHex(key0)}`);
        }
        let fsModule = null;
        let fsEntry = null;
        if (key1 !== 0n) {
          fsEntry = await resolveShaderFromPack(shaderPack, key1);
          if (!fsEntry) {
            throw new Error(`fragment shader ${formatShaderKeyHex(key1)} not found in pack`);
          }
          fsModule = await compileShaderEntry(
            device,
            pipelineCache,
            fsEntry,
            `prewarm-fs-${formatShaderKeyHex(key1)}`,
          );
          if (!fsModule) {
            throw new Error(`failed to compile fragment shader ${formatShaderKeyHex(key1)}`);
          }
        }
        if (pipelineCache && typeof pipelineCache.prewarmGraphicsPipeline === 'function') {
          await pipelineCache.prewarmGraphicsPipeline({
            vsKey: key0,
            fsKey: key1,
            vsModule,
            fsModule,
            vsEntryPoint: vsEntry.entryPoint || 'main',
            fsEntryPoint: fsEntry ? fsEntry.entryPoint || 'main' : undefined,
          });
        }
      }

      const elapsedNs = Math.max(0, Math.round((nowMs() - startMs) * NS_PER_MS));
      return { id: taskId, ok: true, buildNs: elapsedNs };
    } catch (err) {
      const elapsedNs = Math.max(0, Math.round((nowMs() - startMs) * NS_PER_MS));
      return {
        id: taskId,
        ok: false,
        buildNs: elapsedNs,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  function defaultIsProcessorIdle(processor) {
    if (!processor || typeof processor !== 'object') {
      return true;
    }
    if (processor._hasPendingUploadCommands) {
      return false;
    }
    if (Array.isArray(processor._passCommands) && processor._passCommands.length > 0) {
      return false;
    }
    if (typeof processor._pendingShaderCompiles === 'number' && processor._pendingShaderCompiles > 0) {
      return false;
    }
    if (typeof processor._inFlightPipelineCompiles === 'number' && processor._inFlightPipelineCompiles > 0) {
      return false;
    }
    if (
      processor._compileQueue &&
      typeof processor._compileQueue.length === 'number' &&
      processor._compileQueue.length > 0
    ) {
      return false;
    }
    return true;
  }

  /**
   * Clean worker-side message handler that can be attached to `wgpu_worker.js` without invasive edits.
   * Yields behind live title submits/compiles (#2054) and reports `yieldCount` / `yieldMs`.
   */
  function attachPrewarmBridge(processor, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxInFlight = Math.max(
      1,
      Math.min(
        PREWARM_HARD_MAX_IN_FLIGHT,
        Number.isInteger(opts.maxInFlight) ? opts.maxInFlight : PREWARM_DEFAULT_MAX_IN_FLIGHT,
      ),
    );
    const defaultPost =
      typeof opts.postMessage === 'function' ? opts.postMessage : null;
    const isIdle =
      typeof opts.isIdle === 'function' ? opts.isIdle : () => defaultIsProcessorIdle(processor);
    const nowMs = typeof opts.now === 'function' ? opts.now : defaultNowMs;
    const retryDelayMs =
      typeof opts.yieldRetryMs === 'number' && Number.isFinite(opts.yieldRetryMs) && opts.yieldRetryMs >= 1
        ? Math.floor(opts.yieldRetryMs)
        : 16;
    const scheduleTimeout =
      typeof opts.setTimeout === 'function'
        ? opts.setTimeout
        : typeof setTimeout === 'function'
          ? setTimeout
          : null;
    const cancelTimeout =
      typeof opts.clearTimeout === 'function'
        ? opts.clearTimeout
        : typeof clearTimeout === 'function'
          ? clearTimeout
          : null;

    let detached = false;
    let inFlight = 0;
    let peakInFlight = 0;
    let completed = 0;
    let failed = 0;
    let yieldCount = 0;
    let yieldMs = 0;
    let yieldStartedAtMs = null;
    let retryTimer = null;
    const queue = [];

    function closeYieldWindow() {
      if (yieldStartedAtMs !== null) {
        const delta = nowMs() - yieldStartedAtMs;
        if (delta > 0) {
          yieldMs += delta;
        }
        yieldStartedAtMs = null;
      }
    }

    function clearRetryTimer() {
      if (retryTimer !== null && cancelTimeout) {
        try {
          cancelTimeout(retryTimer);
        } catch {
          // Ignore clearTimeout errors.
        }
        retryTimer = null;
      }
    }

    function scheduleRetry() {
      if (detached || retryTimer !== null || !scheduleTimeout) {
        return;
      }
      retryTimer = scheduleTimeout(() => {
        retryTimer = null;
        drain();
      }, retryDelayMs);
      if (retryTimer && typeof retryTimer.unref === 'function') {
        retryTimer.unref();
      }
    }

    function drain() {
      while (!detached && inFlight < maxInFlight && queue.length > 0) {
        let idleNow = true;
        try {
          idleNow = Boolean(isIdle());
        } catch {
          idleNow = false;
        }
        if (!idleNow) {
          yieldCount += 1;
          if (yieldStartedAtMs === null) {
            yieldStartedAtMs = nowMs();
          }
          scheduleRetry();
          return;
        }
        closeYieldWindow();
        clearRetryTimer();
        const item = queue.shift();
        inFlight += 1;
        if (inFlight > peakInFlight) {
          peakInFlight = inFlight;
        }
        executePrewarmTask(processor, item.task, opts)
          .then((result) => {
            inFlight -= 1;
            completed += 1;
            if (!result.ok) {
              failed += 1;
            }
            const reply = item.replyFn || defaultPost;
            if (!detached && typeof reply === 'function') {
              try {
                reply({
                  type: PREWARM_MSG_RESULT,
                  id: result.id,
                  ok: result.ok,
                  buildNs: result.buildNs,
                  error: result.error || null,
                });
              } catch {
                // Non-fatal postMessage reply failure.
              }
            }
            if (typeof item.resolve === 'function') {
              item.resolve(result);
            }
            drain();
          })
          .catch((err) => {
            inFlight -= 1;
            completed += 1;
            failed += 1;
            const fallback = {
              id: Number(item.task && item.task.id) >>> 0,
              ok: false,
              buildNs: 0,
              error: err instanceof Error ? err.message : String(err),
            };
            const reply = item.replyFn || defaultPost;
            if (!detached && typeof reply === 'function') {
              try {
                reply(Object.assign({ type: PREWARM_MSG_RESULT }, fallback));
              } catch {
                // Ignore reply failure.
              }
            }
            if (typeof item.resolve === 'function') {
              item.resolve(fallback);
            }
            drain();
          });
      }
    }

    function executeTask(task, replyFn) {
      if (detached) {
        return Promise.resolve({
          id: Number(task && task.id) >>> 0,
          ok: false,
          buildNs: 0,
          error: 'prewarm bridge detached',
        });
      }
      return new Promise((resolve) => {
        queue.push({ task, replyFn, resolve });
        drain();
      });
    }

    function handleMessage(msg, replyFn) {
      if (detached || !msg || typeof msg !== 'object') {
        return false;
      }
      if (msg.type === PREWARM_MSG_REQUEST && msg.task && typeof msg.task === 'object') {
        executeTask(msg.task, replyFn);
        return true;
      }
      return false;
    }

    return {
      handleMessage,
      executeTask,
      pump: drain,
      getStats() {
        const liveYieldMs =
          yieldStartedAtMs !== null ? yieldMs + Math.max(0, nowMs() - yieldStartedAtMs) : yieldMs;
        return {
          detached,
          maxInFlight,
          inFlight,
          queued: queue.length,
          peakInFlight,
          completed,
          failed,
          yieldCount,
          yieldMs: liveYieldMs,
        };
      },
      detach() {
        detached = true;
        closeYieldWindow();
        clearRetryTimer();
        queue.length = 0;
      },
    };
  }

  /**
   * Loader/coordinator-side prewarm orchestrator bridging the Wasm runtime queue (`#1613`),
   * background shader/pipeline compilation (capped at <= 4 in-flight, idle-gated), and
   * `'PG6W'` warm-list persistence.
   */
  class PrewarmBridge {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      this._module = opts.module || null;
      this._api = resolvePrewarmWasmApi(this._module, opts);
      this._store =
        opts.store instanceof PrewarmListStore
          ? opts.store
          : new PrewarmListStore(opts.storeOptions || (opts.storage ? { storage: opts.storage } : {}));
      this._storeKey = sanitizeStoreKey(opts.storeKey);

      const requestedMax = Number.isInteger(opts.maxInFlight)
        ? opts.maxInFlight
        : PREWARM_DEFAULT_MAX_IN_FLIGHT;
      this._maxInFlight = Math.max(1, Math.min(PREWARM_HARD_MAX_IN_FLIGHT, requestedMax));

      this._pollIntervalMs =
        typeof opts.pollIntervalMs === 'number' &&
        Number.isFinite(opts.pollIntervalMs) &&
        opts.pollIntervalMs >= 1
          ? Math.floor(opts.pollIntervalMs)
          : PREWARM_DEFAULT_POLL_INTERVAL_MS;

      this._now = typeof opts.now === 'function' ? opts.now : defaultNowMs;
      this._isIdle = typeof opts.isIdle === 'function' ? opts.isIdle : null;
      this._buildTask = typeof opts.buildTask === 'function' ? opts.buildTask : null;
      this._gpuWorker = opts.gpuWorker || null;
      this._onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;
      this._setInterval =
        typeof opts.setInterval === 'function'
          ? opts.setInterval
          : typeof setInterval === 'function'
            ? setInterval
            : null;
      this._clearInterval =
        typeof opts.clearInterval === 'function'
          ? opts.clearInterval
          : typeof clearInterval === 'function'
            ? clearInterval
            : null;

      this._inFlight = new Map();
      this._active = false;
      this._destroyed = false;
      this._pollTimer = null;

      this._peakInFlight = 0;
      this._tasksTaken = 0;
      this._tasksCompleted = 0;
      this._tasksSucceeded = 0;
      this._tasksFailed = 0;
      this._yieldCount = 0;
      this._yieldMs = 0;
      this._yieldStartedAtMs = null;
      this._listLoadedCount = 0;
      this._listLoadErrors = 0;
      this._listSaveCount = 0;
      this._lastSavedBytes = 0;
      this._lastError = null;
      this._lastProgress = null;
    }

    get available() {
      return this._api.available;
    }

    get active() {
      return this._active && !this._destroyed;
    }

    get maxInFlight() {
      return this._maxInFlight;
    }

    get inFlightCount() {
      return this._inFlight.size;
    }

    get store() {
      return this._store;
    }

    setGpuWorker(worker) {
      this._gpuWorker = worker || null;
    }

    /**
     * Validate and pass a `'PG6W'` warm list into `playgta6_prewarm_load_list(ptr, size)`.
     */
    loadWarmList(bytes) {
      if (this._destroyed || !this._api.hasLoadList) {
        return { ok: false, queued: -1, byteLength: 0, error: 'prewarm load export unavailable' };
      }
      const view = toUint8View(bytes);
      const parsed = parseWarmList(view);
      if (!parsed.ok) {
        this._listLoadErrors += 1;
        this._lastError = parsed.error;
        return { ok: false, queued: -1, byteLength: view ? view.byteLength : 0, error: parsed.error };
      }
      try {
        const queued = this._api.loadList(view);
        if (queued < 0) {
          this._listLoadErrors += 1;
          this._lastError = 'runtime rejected warm list';
          return {
            ok: false,
            queued: -1,
            byteLength: view.byteLength,
            error: 'runtime rejected warm list',
          };
        }
        this._listLoadedCount += queued;
        this.readProgress();
        return {
          ok: true,
          queued,
          version: parsed.version,
          entryCount: parsed.count,
          byteLength: view.byteLength,
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this._listLoadErrors += 1;
        this._lastError = message;
        return { ok: false, queued: -1, byteLength: view.byteLength, error: message };
      }
    }

    /**
     * Load the persisted `'PG6W'` warm list from OPFS/IndexedDB/memory and feed it to Wasm.
     */
    async loadPersistedList(key = this._storeKey) {
      if (this._destroyed || !this._api.hasLoadList) {
        return { ok: false, queued: 0, found: false, backend: 'none' };
      }
      try {
        const bytes = await this._store.load(key);
        if (!bytes) {
          return {
            ok: true,
            queued: 0,
            found: false,
            backend: this._store.lastBackend,
          };
        }
        const res = this.loadWarmList(bytes);
        return Object.assign({ found: true, backend: this._store.lastBackend }, res);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this._listLoadErrors += 1;
        this._lastError = message;
        return { ok: false, queued: -1, found: false, backend: this._store.lastBackend, error: message };
      }
    }

    /**
     * Read the 48-byte `PrewarmProgress` struct via `playgta6_prewarm_progress(out)`.
     */
    readProgress() {
      if (this._destroyed || !this._api.hasProgress) {
        return this._lastProgress ? Object.assign({}, this._lastProgress) : null;
      }
      try {
        const progress = this._api.progress();
        if (progress) {
          this._lastProgress = progress;
          if (this._onProgress) {
            try {
              this._onProgress(Object.assign({}, progress));
            } catch {
              // Progress observer errors must never break prewarm execution.
            }
          }
          return Object.assign({}, progress);
        }
      } catch (err) {
        this._lastError = err instanceof Error ? err.message : String(err);
      }
      return this._lastProgress ? Object.assign({}, this._lastProgress) : null;
    }

    /**
     * Complete an in-flight task (called from local `buildTask` promise or worker reply message).
     */
    _finishTask(id, ok, buildNs, errorMsg) {
      const taskId = Number(id) >>> 0;
      const entry = this._inFlight.get(taskId);
      if (!entry) {
        return false;
      }
      this._inFlight.delete(taskId);

      const finalNs =
        typeof buildNs === 'number' && Number.isFinite(buildNs) && buildNs >= 0
          ? Math.round(buildNs)
          : Math.max(0, Math.round((this._now() - entry.startedAtMs) * NS_PER_MS));

      let accepted = false;
      if (!this._destroyed && this._api.hasComplete) {
        try {
          accepted = this._api.complete(taskId, Boolean(ok), finalNs);
        } catch (err) {
          this._lastError = err instanceof Error ? err.message : String(err);
        }
      }

      this._tasksCompleted += 1;
      if (ok) {
        this._tasksSucceeded += 1;
      } else {
        this._tasksFailed += 1;
        if (errorMsg) {
          this._lastError = String(errorMsg);
        }
      }

      if (!this._destroyed && this._api.hasProgress) {
        this.readProgress();
      }
      if (this._active && !this._destroyed) {
        this.pump();
      }
      return accepted;
    }

    /**
     * Handle a `prewarm-task-result` message returned from `wgpu_worker.js`.
     */
    handleWorkerMessage(msg) {
      if (!msg || typeof msg !== 'object' || msg.type !== PREWARM_MSG_RESULT) {
        return false;
      }
      this._finishTask(msg.id, Boolean(msg.ok), msg.buildNs, msg.error);
      return true;
    }

    /**
     * Poll `playgta6_prewarm_take(out, availableSlots)` and dispatch background builds while
     * respecting idle gating and the concurrency cap (`<= 4`).
     *
     * @returns {number} Number of newly dispatched tasks.
     */
    pump() {
      if (this._destroyed || !this._api.hasTake) {
        return 0;
      }
      if (!this._buildTask && !(this._gpuWorker && typeof this._gpuWorker.postMessage === 'function')) {
        return 0;
      }
      if (this._isIdle) {
        let idleNow = false;
        try {
          idleNow = Boolean(this._isIdle());
        } catch {
          idleNow = false;
        }
        if (!idleNow) {
          this._yieldCount += 1;
          if (this._yieldStartedAtMs === null) {
            this._yieldStartedAtMs = this._now();
          }
          return 0;
        }
        if (this._yieldStartedAtMs !== null) {
          const delta = this._now() - this._yieldStartedAtMs;
          if (delta > 0) {
            this._yieldMs += delta;
          }
          this._yieldStartedAtMs = null;
        }
      }

      const availableSlots = this._maxInFlight - this._inFlight.size;
      if (availableSlots <= 0) {
        return 0;
      }

      let tasks = [];
      try {
        tasks = this._api.take(availableSlots);
      } catch (err) {
        this._lastError = err instanceof Error ? err.message : String(err);
        return 0;
      }

      if (!Array.isArray(tasks) || tasks.length === 0) {
        return 0;
      }

      let dispatched = 0;
      for (let i = 0; i < tasks.length; i += 1) {
        if (this._inFlight.size >= this._maxInFlight) {
          break;
        }
        const task = tasks[i];
        if (!task || !task.id || this._inFlight.has(task.id)) {
          continue;
        }

        const startedAtMs = this._now();
        this._inFlight.set(task.id, { task, startedAtMs });
        this._tasksTaken += 1;
        dispatched += 1;
        if (this._inFlight.size > this._peakInFlight) {
          this._peakInFlight = this._inFlight.size;
        }

        if (this._buildTask) {
          Promise.resolve()
            .then(() => this._buildTask(task))
            .then((res) => {
              const elapsedNs = Math.max(0, Math.round((this._now() - startedAtMs) * NS_PER_MS));
              if (typeof res === 'boolean') {
                this._finishTask(task.id, res, elapsedNs, res ? null : 'prewarm task returned false');
                return;
              }
              if (res && typeof res === 'object') {
                const ok = res.ok !== false;
                const ns =
                  typeof res.buildNs === 'number' && Number.isFinite(res.buildNs) && res.buildNs >= 0
                    ? Math.round(res.buildNs)
                    : elapsedNs;
                this._finishTask(task.id, ok, ns, ok ? null : res.error || 'prewarm build failed');
                return;
              }
              this._finishTask(task.id, true, elapsedNs, null);
            })
            .catch((err) => {
              const elapsedNs = Math.max(0, Math.round((this._now() - startedAtMs) * NS_PER_MS));
              this._finishTask(
                task.id,
                false,
                elapsedNs,
                err instanceof Error ? err.message : String(err),
              );
            });
        } else if (this._gpuWorker && typeof this._gpuWorker.postMessage === 'function') {
          try {
            this._gpuWorker.postMessage({
              type: PREWARM_MSG_REQUEST,
              task: {
                id: task.id,
                kind: task.kind,
                kindName: kindToString(task.kind),
                key0: formatShaderKeyHex(task.key0),
                key1: formatShaderKeyHex(task.key1),
              },
            });
          } catch (err) {
            const elapsedNs = Math.max(0, Math.round((this._now() - startedAtMs) * NS_PER_MS));
            this._finishTask(
              task.id,
              false,
              elapsedNs,
              err instanceof Error ? err.message : String(err),
            );
          }
        }
      }

      return dispatched;
    }

    /**
     * Drain all currently queued tasks to completion (respecting `maxInFlight <= 4`).
     * Useful for deterministic tests and loading-phase prewarm barriers.
     */
    async drainAll({ maxIterations = 10000 } = {}) {
      let iterations = 0;
      while (!this._destroyed && iterations < maxIterations) {
        iterations += 1;
        const dispatched = this.pump();
        if (dispatched === 0 && this._inFlight.size === 0) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      return this.readProgress();
    }

    /**
     * Export the observed `'PG6W'` warm list via `playgta6_prewarm_save_list(out, cap)`
     * and optionally persist it to IndexedDB/OPFS/memory.
     */
    async saveWarmList({ persist = true, key = this._storeKey } = {}) {
      if (this._destroyed || !this._api.hasSaveList) {
        return {
          ok: false,
          bytes: null,
          byteLength: 0,
          entryCount: 0,
          version: 0,
          persisted: false,
          error: 'prewarm save export unavailable',
        };
      }
      try {
        const bytes = this._api.saveList();
        if (!bytes || bytes.byteLength === 0) {
          return {
            ok: true,
            bytes: null,
            byteLength: 0,
            entryCount: 0,
            version: 0,
            persisted: false,
          };
        }
        const parsed = parseWarmList(bytes);
        if (!parsed.ok) {
          this._lastError = parsed.error;
          return {
            ok: false,
            bytes: null,
            byteLength: bytes.byteLength,
            entryCount: 0,
            version: 0,
            persisted: false,
            error: parsed.error,
          };
        }

        let persisted = false;
        let backend = 'none';
        if (persist) {
          const saveRes = await this._store.save(key, bytes);
          persisted = Boolean(saveRes.ok);
          backend = saveRes.backend || 'none';
          if (!saveRes.ok && saveRes.error) {
            this._lastError = saveRes.error;
          }
        }

        this._listSaveCount += 1;
        this._lastSavedBytes = bytes.byteLength;
        return {
          ok: true,
          bytes,
          byteLength: bytes.byteLength,
          entryCount: parsed.count,
          version: parsed.version,
          persisted,
          backend,
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this._lastError = message;
        return {
          ok: false,
          bytes: null,
          byteLength: 0,
          entryCount: 0,
          version: 0,
          persisted: false,
          error: message,
        };
      }
    }

    start() {
      if (this._destroyed || !this._api.available) {
        return false;
      }
      if (this._active) {
        return true;
      }
      this._active = true;
      this.pump();
      if (this._setInterval && this._pollTimer === null) {
        this._pollTimer = this._setInterval(() => {
          if (!this._active || this._destroyed) {
            return;
          }
          this.pump();
        }, this._pollIntervalMs);
        if (this._pollTimer && typeof this._pollTimer.unref === 'function') {
          this._pollTimer.unref();
        }
      }
      return true;
    }

    async stop({ save = false } = {}) {
      this._active = false;
      if (this._pollTimer !== null && this._clearInterval) {
        try {
          this._clearInterval(this._pollTimer);
        } catch {
          // Ignore clearInterval error.
        }
        this._pollTimer = null;
      }
      if (save && !this._destroyed && this._api.hasSaveList) {
        return this.saveWarmList({ persist: true });
      }
      return null;
    }

    destroy() {
      this._active = false;
      this._destroyed = true;
      if (this._yieldStartedAtMs !== null) {
        const delta = this._now() - this._yieldStartedAtMs;
        if (delta > 0) {
          this._yieldMs += delta;
        }
        this._yieldStartedAtMs = null;
      }
      if (this._pollTimer !== null && this._clearInterval) {
        try {
          this._clearInterval(this._pollTimer);
        } catch {
          // Ignore clearInterval error.
        }
        this._pollTimer = null;
      }
      this._inFlight.clear();
    }

    getDiagnostics() {
      const progress = this._destroyed
        ? this._lastProgress
          ? Object.assign({}, this._lastProgress)
          : null
        : this.readProgress();
      const liveYieldMs =
        this._yieldStartedAtMs !== null
          ? this._yieldMs + Math.max(0, this._now() - this._yieldStartedAtMs)
          : this._yieldMs;
      return {
        available: this._api.available,
        active: this.active,
        maxInFlight: this._maxInFlight,
        inFlightCount: this._inFlight.size,
        peakInFlight: this._peakInFlight,
        tasksTaken: this._tasksTaken,
        tasksCompleted: this._tasksCompleted,
        tasksSucceeded: this._tasksSucceeded,
        tasksFailed: this._tasksFailed,
        yieldCount: this._yieldCount,
        yieldMs: liveYieldMs,
        listLoadedCount: this._listLoadedCount,
        listLoadErrors: this._listLoadErrors,
        listSaveCount: this._listSaveCount,
        lastSavedBytes: this._lastSavedBytes,
        storageBackend: this._store.lastBackend,
        corruptStorageReads: this._store.corruptReads,
        lastError: this._lastError,
        progress,
      };
    }
  }

  const exportedApi = {
    PREWARM_MAGIC,
    PREWARM_MAGIC_BYTES,
    PREWARM_LIST_VERSION_MODULES,
    PREWARM_LIST_VERSION_PIPELINES,
    PREWARM_LIST_HEADER_BYTES,
    PREWARM_LIST_ENTRY_BYTES,
    PREWARM_TASK_BYTES,
    PREWARM_PROGRESS_BYTES,
    PREWARM_KIND_GRAPHICS,
    PREWARM_KIND_COMPUTE,
    PREWARM_DEFAULT_MAX_IN_FLIGHT,
    PREWARM_HARD_MAX_IN_FLIGHT,
    PREWARM_MAX_ENTRIES,
    PREWARM_MAX_DETAIL_BYTES,
    PREWARM_DEFAULT_POLL_INTERVAL_MS,
    PREWARM_DEFAULT_SAVE_INTERVAL_MS,
    PREWARM_DEFAULT_STORE_KEY,
    PREWARM_DEFAULT_DB_NAME,
    PREWARM_DEFAULT_STORE_NAME,
    PREWARM_DEFAULT_OPFS_DIR,
    PREWARM_MSG_REQUEST,
    PREWARM_MSG_RESULT,
    PREWARM_MSG_PROGRESS,
    PREWARM_MSG_SAVE,
    formatShaderKeyHex,
    parseWarmList,
    serializeWarmList,
    decodePrewarmTasks,
    decodePrewarmProgress,
    resolvePrewarmWasmApi,
    PrewarmListStore,
    PrewarmBridge,
    executePrewarmTask,
    attachPrewarmBridge,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PrewarmBridge = exportedApi;
  } else if (root) {
    root.PlayGta6PrewarmBridge = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
