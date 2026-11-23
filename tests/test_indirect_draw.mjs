import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const mod = require('../web/workers/indirect_draw.js');

const {
  U32_MAX,
  BUFFER_USAGE,
  INDIRECT_KIND,
  DRAW_INDIRECT_LAYOUT,
  DRAW_INDEXED_INDIRECT_LAYOUT,
  DISPATCH_INDIRECT_LAYOUT,
  IndirectDrawError,
  recordByteSize,
  wgslStructSource,
  wgslArrayStride,
  encodeDrawIndirect,
  encodeDrawIndexedIndirect,
  encodeDispatchIndirect,
  decodeDrawIndirect,
  decodeDrawIndexedIndirect,
  decodeDispatchIndirect,
  encodeBatch,
  validateIndirectRange,
  validateDispatchCounts,
  planDispatches,
  encodeDispatches,
  encodeDispatchForItems,
  encodeMultiDraw,
  IndirectArgsBuffer,
} = mod;

function createMockDevice() {
  const buffers = [];
  const writes = [];
  const submits = [];
  const copies = [];
  const queue = {
    writeBuffer(buffer, offset, data) {
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
      buffer._data.set(bytes, offset);
      writes.push({ buffer, offset, length: bytes.byteLength });
    },
    submit(cmds) {
      submits.push(cmds);
      for (const cmd of cmds) {
        for (const c of cmd.copies) {
          c.dst._data.set(c.src._data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
        }
      }
    },
  };
  const device = {
    queue,
    createBuffer(desc) {
      const buffer = {
        ...desc,
        _data: new Uint8Array(desc.size),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      const mine = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          const c = { src, srcOffset, dst, dstOffset, size };
          mine.push(c);
          copies.push(c);
        },
        finish() {
          return { copies: mine };
        },
      };
    },
  };
  return { device, buffers, writes, submits, copies };
}

function createMockPass(withNative = false) {
  const calls = [];
  const pass = {
    calls,
    drawIndirect: (b, o) => calls.push(['drawIndirect', o]),
    drawIndexedIndirect: (b, o) => calls.push(['drawIndexedIndirect', o]),
    dispatchWorkgroups: (x, y, z) => calls.push(['dispatchWorkgroups', x, y, z]),
  };
  if (withNative) {
    pass.multiDrawIndirect = (b, o, n) => calls.push(['multiDrawIndirect', o, n]);
    pass.multiDrawIndexedIndirect = (b, o, n) => calls.push(['multiDrawIndexedIndirect', o, n]);
  }
  return pass;
}

const rec = (n) => ({ indexCount: n, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 });

test('layout sizes are 16 / 20 / 12 bytes', () => {
  assert.equal(DRAW_INDIRECT_LAYOUT.byteSize, 16);
  assert.equal(DRAW_INDEXED_INDIRECT_LAYOUT.byteSize, 20);
  assert.equal(DISPATCH_INDIRECT_LAYOUT.byteSize, 12);
  assert.equal(recordByteSize(INDIRECT_KIND.DRAW_INDEXED), 20);
  assert.throws(() => recordByteSize('nope'), TypeError);
});

test('drawIndirect encodes byte-exact little-endian', () => {
  const bytes = encodeDrawIndirect({ vertexCount: 3, instanceCount: 0x01020304, firstVertex: 7, firstInstance: 0 });
  assert.deepEqual(
    [...bytes],
    [3, 0, 0, 0, 4, 3, 2, 1, 7, 0, 0, 0, 0, 0, 0, 0]
  );
});

test('drawIndexedIndirect encodes negative baseVertex and max u32 byte-exact', () => {
  const bytes = encodeDrawIndexedIndirect({
    indexCount: U32_MAX,
    instanceCount: 1,
    firstIndex: 2,
    baseVertex: -1,
    firstInstance: U32_MAX,
  });
  assert.equal(bytes.byteLength, 20);
  assert.deepEqual(
    [...bytes],
    [255, 255, 255, 255, 1, 0, 0, 0, 2, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]
  );
  const min = encodeDrawIndexedIndirect({ indexCount: 0, instanceCount: 0, baseVertex: -0x80000000 });
  assert.deepEqual([...min.subarray(12, 16)], [0, 0, 0, 0x80]);
});

test('dispatchWorkgroupsIndirect encodes byte-exact', () => {
  assert.deepEqual([...encodeDispatchIndirect({ x: 1, y: 2, z: 0x10000 })], [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0]);
});

test('round-trip decode for all three layouts', () => {
  const d = { vertexCount: 36, instanceCount: 5, firstVertex: 9, firstInstance: 11 };
  assert.deepEqual(decodeDrawIndirect(encodeDrawIndirect(d)), d);
  const i = { indexCount: U32_MAX, instanceCount: 2, firstIndex: 100, baseVertex: -12345, firstInstance: 3 };
  assert.deepEqual(decodeDrawIndexedIndirect(encodeDrawIndexedIndirect(i)), i);
  const c = { x: U32_MAX, y: 1, z: 7 };
  assert.deepEqual(decodeDispatchIndirect(encodeDispatchIndirect(c)), c);
});

test('encode/decode work on ArrayBuffer, DataView and offset typed arrays', () => {
  const ab = new ArrayBuffer(48);
  mod.encodeRecord(INDIRECT_KIND.DRAW_INDEXED, rec(9), new DataView(ab), 24);
  assert.equal(decodeDrawIndexedIndirect(ab, 24).indexCount, 9);
  const u32 = new Uint32Array(ab, 4, 11);
  assert.equal(decodeDrawIndexedIndirect(u32, 20).indexCount, 9);
});

test('validation: non-integer field rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: 1.5, instanceCount: 1 }), TypeError);
});
test('validation: missing required count rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: 1 }), TypeError);
});
test('validation: u32 overflow rejected', () => {
  assert.throws(() => encodeDrawIndirect({ vertexCount: U32_MAX + 1, instanceCount: 1 }), RangeError);
});
test('validation: negative u32 rejected', () => {
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: -1, instanceCount: 1 }), RangeError);
});
test('validation: baseVertex outside i32 rejected', () => {
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: 1, instanceCount: 1, baseVertex: 0x80000000 }), RangeError);
  assert.throws(() => encodeDrawIndexedIndirect({ indexCount: 1, instanceCount: 1, baseVertex: -0x80000001 }), RangeError);
});
test('validation: undersized target rejected', () => {
  assert.throws(() => mod.encodeRecord(INDIRECT_KIND.DRAW, { vertexCount: 1, instanceCount: 1 }, new Uint8Array(15)), RangeError);
});

test('encodeBatch writes records at a stride', () => {
  const { bytes, stride, patched } = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), rec(2), rec(3)], { stride: 32 });
  assert.equal(stride, 32);
  assert.deepEqual(patched, []);
  assert.equal(bytes.byteLength, 32 * 2 + 20);
  assert.equal(decodeDrawIndexedIndirect(bytes, 64).indexCount, 3);
  assert.equal(decodeDrawIndexedIndirect(bytes, 32).indexCount, 2);
});

test('encodeBatch rejects bad strides', () => {
  assert.throws(() => encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1)], { stride: 16 }), RangeError);
  assert.throws(() => encodeBatch(INDIRECT_KIND.DRAW, [{ vertexCount: 1, instanceCount: 1 }], { stride: 18 }), RangeError);
});

test('firstInstance without feature throws FIRST_INSTANCE_UNSUPPORTED', () => {
  assert.throws(
    () => encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(1), { ...rec(2), firstInstance: 4 }]),
    (e) => e instanceof IndirectDrawError && e.code === 'FIRST_INSTANCE_UNSUPPORTED' && /record 1/.test(e.message)
  );
});

test('firstInstance allowed with feature flag', () => {
  const { bytes, patched } = encodeBatch(INDIRECT_KIND.DRAW, [{ vertexCount: 3, instanceCount: 1, firstInstance: 4 }], {
    indirectFirstInstance: true,
  });
  assert.equal(decodeDrawIndirect(bytes).firstInstance, 4);
  assert.deepEqual(patched, []);
});

test('firstInstance patch fallback zeroes and reports indices', () => {
  const { bytes, patched } = encodeBatch(
    INDIRECT_KIND.DRAW_INDEXED,
    [{ ...rec(1), firstInstance: 9 }, rec(2)],
    { patchFirstInstance: true }
  );
  assert.deepEqual(patched, [0]);
  assert.equal(decodeDrawIndexedIndirect(bytes, 0).firstInstance, 0);
});

test('encodeBatch can write into a caller buffer at an offset', () => {
  const out = new Uint8Array(64);
  encodeBatch(INDIRECT_KIND.DRAW_INDEXED, [rec(5)], { out, outOffset: 40 });
  assert.equal(decodeDrawIndexedIndirect(out, 40).indexCount, 5);
});

test('validateIndirectRange: missing INDIRECT usage', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.STORAGE, size: 64 }, 0, 16),
    (e) => e.code === 'NOT_INDIRECT'
  );
});
test('validateIndirectRange: offset not multiple of 4', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 2, 16),
    (e) => e.code === 'MISALIGNED_OFFSET'
  );
});
test('validateIndirectRange: offset + record beyond buffer', () => {
  assert.throws(
    () => validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 48, 20),
    (e) => e.code === 'OUT_OF_BOUNDS'
  );
});
test('validateIndirectRange: exact fit accepted', () => {
  validateIndirectRange({ usage: BUFFER_USAGE.INDIRECT, size: 64 }, 44, 20);
});

test('validateDispatchCounts: limit enforced per axis', () => {
  assert.deepEqual(validateDispatchCounts({ x: 65535 }), { x: 65535, y: 1, z: 1 });
  for (const axis of ['x', 'y', 'z']) {
    assert.throws(
      () => validateDispatchCounts({ x: 1, y: 1, z: 1, [axis]: 65536 }),
      (e) => e.code === 'DISPATCH_LIMIT' && e.message.includes(`${axis}=65536`)
    );
  }
  assert.throws(() => validateDispatchCounts({ x: 11 }, 10), (e) => e.code === 'DISPATCH_LIMIT');
  assert.throws(() => validateDispatchCounts({ x: -1 }), RangeError);
});

test('planDispatches: fits one dispatch at exactly the limit', () => {
  const plan = planDispatches(64 * 10, 64, 10);
  assert.deepEqual(plan, [{ x: 10, y: 1, z: 1, firstWorkgroup: 0, firstItem: 0 }]);
});

test('planDispatches: one over the limit splits in two', () => {
  const plan = planDispatches(64 * 10 + 1, 64, 10);
  assert.equal(plan.length, 2);
  assert.deepEqual(plan[0], { x: 10, y: 1, z: 1, firstWorkgroup: 0, firstItem: 0 });
  assert.deepEqual(plan[1], { x: 1, y: 1, z: 1, firstWorkgroup: 10, firstItem: 640 });
});

test('planDispatches: rounding and totals at default limit', () => {
  assert.deepEqual(planDispatches(0, 64), []);
  assert.deepEqual(planDispatches(1, 64).map((d) => d.x), [1]);
  const items = 65535 * 64 * 2 + 5;
  const plan = planDispatches(items, 64);
  assert.deepEqual(plan.map((d) => d.x), [65535, 65535, 1]);
  assert.equal(plan.reduce((s, d) => s + d.x, 0), Math.ceil(items / 64));
  assert.throws(() => planDispatches(1, 0), RangeError);
});

test('encodeDispatches calls dispatchWorkgroups with hook', () => {
  const pass = createMockPass();
  const seen = [];
  const n = encodeDispatches(pass, planDispatches(25, 1, 10), (d, i) => seen.push([i, d.firstItem]));
  assert.equal(n, 3);
  assert.deepEqual(pass.calls, [
    ['dispatchWorkgroups', 10, 1, 1],
    ['dispatchWorkgroups', 10, 1, 1],
    ['dispatchWorkgroups', 5, 1, 1],
  ]);
  assert.deepEqual(seen, [[0, 0], [1, 10], [2, 20]]);
});

test('encodeDispatchForItems encodes an indirect record and enforces the limit', () => {
  assert.deepEqual(decodeDispatchIndirect(encodeDispatchForItems(1000, 64)), { x: 16, y: 1, z: 1 });
  assert.throws(() => encodeDispatchForItems(64 * 11, 64, 10), (e) => e.code === 'DISPATCH_LIMIT');
});

test('WGSL struct layout offsets and strides', () => {
  assert.equal(wgslArrayStride(INDIRECT_KIND.DRAW), 16);
  assert.equal(wgslArrayStride(INDIRECT_KIND.DRAW_INDEXED), 20);
  assert.equal(wgslArrayStride(INDIRECT_KIND.DISPATCH), 12);
  // Member offsets follow declaration order in the WGSL struct source.
  for (const [kind, layout] of [
    [INDIRECT_KIND.DRAW, DRAW_INDIRECT_LAYOUT],
    [INDIRECT_KIND.DRAW_INDEXED, DRAW_INDEXED_INDIRECT_LAYOUT],
    [INDIRECT_KIND.DISPATCH, DISPATCH_INDIRECT_LAYOUT],
  ]) {
    const members = [...wgslStructSource(kind).matchAll(/^\s+(\w+): (u32|i32),$/gm)];
    assert.equal(members.length * 4, layout.byteSize);
    const camel = (s) => s.replace(/_(\w)/g, (_, c) => c.toUpperCase());
    members.forEach((m, idx) => {
      const field = layout.fields[camel(m[1])];
      assert.ok(field, `${kind}.${m[1]} maps to a field`);
      assert.equal(field.offset, idx * 4);
      assert.equal(field.signed, m[2] === 'i32');
    });
  }
  assert.match(wgslStructSource(INDIRECT_KIND.DRAW_INDEXED), /base_vertex: i32/);
  assert.equal(DRAW_INDEXED_INDIRECT_LAYOUT.fields.instanceCount.offset, 4);
  // args[7].instance_count byte offset for the 20-byte indexed record
  assert.equal(7 * wgslArrayStride(INDIRECT_KIND.DRAW_INDEXED) + DRAW_INDEXED_INDIRECT_LAYOUT.fields.instanceCount.offset, 144);
});

test('IndirectArgsBuffer allocates GPU buffer with INDIRECT|STORAGE|COPY_DST|COPY_SRC', () => {
  const { device, buffers } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  assert.equal(buffers.length, 1);
  assert.equal(buffers[0].size, 80);
  assert.equal(
    buffers[0].usage,
    BUFFER_USAGE.INDIRECT | BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC
  );
  const noSrc = new IndirectArgsBuffer({ device, capacity: 4, copySrc: false });
  assert.equal(noSrc.buffer.usage & BUFFER_USAGE.COPY_SRC, 0);
  assert.throws(() => noSrc.grow(8), (e) => e.code === 'GROW_UNSUPPORTED');
  assert.equal(args.getStorageBinding().size, 80);
});

test('IndirectArgsBuffer slot offsets are stable and freed slots are reused', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  const a = args.allocate(rec(1));
  const b = args.allocate(rec(2));
  const c = args.allocate(rec(3));
  assert.deepEqual([a, b, c], [0, 1, 2]);
  assert.equal(args.slotOffset(c), 40);
  args.free(b);
  assert.equal(args.liveCount, 2);
  assert.equal(args.allocate(rec(9)), b);
  assert.equal(args.getArgs(b).indexCount, 9);
  assert.throws(() => args.free(3), (e) => e.code === 'BAD_SLOT');
  args.free(a);
  assert.throws(() => args.free(a), (e) => e.code === 'BAD_SLOT');
});

test('IndirectArgsBuffer reused slot without args is zeroed', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = args.allocate(rec(5));
  args.free(s);
  assert.equal(args.allocate(), s);
  assert.equal(args.getArgs(s).indexCount, 0);
});

test('dirty-range coalescing: adjacent slots merge into one write', () => {
  const { device, writes } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 8 });
  for (let i = 0; i < 3; i += 1) args.allocate(rec(i + 1));
  assert.equal(args.flush(), 1);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 60]]);
  assert.equal(args.flush(), 0);
});

test('dirty-range coalescing: gaps stay separate unless mergeGap covers them', () => {
  const { device, writes } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 8 });
  for (let i = 0; i < 6; i += 1) args.allocate(rec(i));
  args.flush();
  writes.length = 0;
  args.setArgs(0, rec(10));
  args.setArgs(2, rec(12));
  args.setArgs(5, rec(15));
  args.setArgs(2, rec(13));
  assert.equal(args.flush(), 3);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 20], [40, 20], [100, 20]]);

  const gap = new IndirectArgsBuffer({ device, capacity: 8, mergeGap: 20 });
  for (let i = 0; i < 6; i += 1) gap.allocate(rec(i));
  gap.flush();
  writes.length = 0;
  gap.setArgs(0, rec(10));
  gap.setArgs(2, rec(12));
  assert.equal(gap.flush(), 1);
  assert.deepEqual(writes.map((w) => [w.offset, w.length]), [[0, 60]]);
});

test('flush uploads exact bytes into the GPU buffer', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = args.allocate({ ...rec(7), baseVertex: -3 });
  args.flush();
  assert.deepEqual(decodeDrawIndexedIndirect(args.buffer._data, args.slotOffset(s)), { ...rec(7), baseVertex: -3 });
});

test('growth preserves data (including GPU-side edits), notifies, and keeps slot indices', () => {
  const { device, buffers, submits, copies } = createMockDevice();
  const events = [];
  const args = new IndirectArgsBuffer({ device, capacity: 2, onGrow: (e) => events.push(e) });
  args.allocate(rec(1));
  args.allocate(rec(2));
  args.flush();
  // Simulate the culling shader zeroing instanceCount of slot 1 on the GPU.
  args.buffer._data.set([0, 0, 0, 0], 20 + 4);
  const old = args.buffer;
  const third = args.allocate(rec(3));
  assert.equal(third, 2);
  assert.equal(args.capacity, 4);
  assert.equal(buffers.length, 2);
  assert.equal(old.destroyed, true);
  assert.equal(submits.length, 1);
  assert.equal(copies.length, 1);
  assert.equal(events.length, 1);
  assert.equal(events[0].oldBuffer, old);
  assert.equal(events[0].buffer, args.buffer);
  assert.equal(events[0].capacity, 4);
  assert.equal(events[0].oldCapacity, 2);
  args.flush();
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 0).indexCount, 1);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 20).indexCount, 2);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 20).instanceCount, 0, "GPU-side cull survives growth");
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 40).indexCount, 3);
  assert.equal(args.getStorageBinding().buffer, args.buffer);
});

test('growth copies GPU-side edits for clean slots', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  args.allocate(rec(1));
  args.allocate(rec(2));
  args.flush();
  args.buffer._data.set([0, 0, 0, 0], 4);
  args.grow(3);
  assert.equal(decodeDrawIndexedIndirect(args.buffer._data, 0).instanceCount, 0);
  assert.equal(args.grow(2), false);
});

test('autoGrow false throws FULL', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 1, autoGrow: false });
  args.allocate();
  assert.throws(() => args.allocate(), (e) => e.code === 'FULL');
});

test('IndirectArgsBuffer applies firstInstance policy', () => {
  const { device } = createMockDevice();
  const strict = new IndirectArgsBuffer({ device, capacity: 2 });
  const s = strict.allocate();
  assert.throws(() => strict.setArgs(s, { ...rec(1), firstInstance: 2 }), (e) => e.code === 'FIRST_INSTANCE_UNSUPPORTED');
  const patching = new IndirectArgsBuffer({ device, capacity: 2, patchFirstInstance: true });
  const p = patching.allocate({ ...rec(1), firstInstance: 2 });
  assert.equal(patching.getArgs(p).firstInstance, 0);
  const native = new IndirectArgsBuffer({ device, capacity: 2, indirectFirstInstance: true });
  assert.equal(native.getArgs(native.allocate({ ...rec(1), firstInstance: 2 })).firstInstance, 2);
});

test('storage binding exposes buffer, size, stride and instanceCount offset', () => {
  const { device } = createMockDevice();
  const idx = new IndirectArgsBuffer({ device, capacity: 3 }).getStorageBinding();
  assert.equal(idx.offset, 0);
  assert.equal(idx.size, 60);
  assert.equal(idx.stride, 20);
  assert.equal(idx.instanceCountOffset, 4);
  assert.deepEqual(idx.bindGroupLayoutEntry, { buffer: { type: 'storage' } });
  assert.match(idx.wgsl, /struct DrawIndexedIndirectArgs/);
  const plain = new IndirectArgsBuffer({ device, kind: INDIRECT_KIND.DRAW, capacity: 3 }).getStorageBinding();
  assert.equal(plain.size, 48);
  assert.equal(plain.stride, 16);
  assert.throws(() => new IndirectArgsBuffer({ device, kind: INDIRECT_KIND.DISPATCH }), TypeError);
});

test('destroy releases the buffer and blocks further use', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 2 });
  args.destroy();
  args.destroy();
  assert.equal(args.buffer.destroyed, true);
  assert.throws(() => args.allocate(), (e) => e.code === 'DESTROYED');
  assert.throws(() => args.flush(), (e) => e.code === 'DESTROYED');
});

const indirectBuf = (size) => ({ usage: BUFFER_USAGE.INDIRECT, size });

test('multi-draw emulation issues one drawIndirect per slot', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(80), { count: 3, firstSlot: 1 });
  assert.deepEqual(pass.calls, [
    ['drawIndirect', 16],
    ['drawIndirect', 32],
    ['drawIndirect', 48],
  ]);
  assert.equal(r.calls, 3);
});

test('multi-draw emulation: exact call sequence without mask (indexed)', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(100), { indexed: true, count: 3, firstSlot: 1, baseOffset: 0 });
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 20],
    ['drawIndexedIndirect', 40],
    ['drawIndexedIndirect', 60],
  ]);
  assert.deepEqual(r, { calls: 3, draws: 3, skipped: 0, native: false });
});

test('multi-draw emulation: visibility mask skips draws', () => {
  const pass = createMockPass();
  const r = encodeMultiDraw(pass, indirectBuf(100), {
    indexed: true,
    count: 5,
    visibility: [true, false, 1, 0, true],
  });
  assert.deepEqual(pass.calls, [
    ['drawIndexedIndirect', 0],
    ['drawIndexedIndirect', 40],
    ['drawIndexedIndirect', 80],
  ]);
  assert.deepEqual(r, { calls: 3, draws: 3, skipped: 2, native: false });
});

test('multi-draw validates every offset against the buffer', () => {
  const pass = createMockPass();
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(30), { indexed: true, count: 2 }), (e) => e.code === 'OUT_OF_BOUNDS');
  assert.throws(() => encodeMultiDraw(pass, { usage: 0, size: 100 }, { count: 1 }), (e) => e.code === 'NOT_INDIRECT');
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(100), { count: 1, baseOffset: 2 }), (e) => e.code === 'MISALIGNED_OFFSET');
  assert.throws(() => encodeMultiDraw(pass, indirectBuf(100), { count: 1, stride: 8 }), RangeError);
});

test('native multi-draw is used for contiguous visible runs only when requested', () => {
  const pass = createMockPass(true);
  const off = encodeMultiDraw(pass, indirectBuf(200), { indexed: true, count: 2 });
  assert.equal(off.native, false);
  assert.deepEqual(pass.calls.map((c) => c[0]), ['drawIndexedIndirect', 'drawIndexedIndirect']);

  pass.calls.length = 0;
  const r = encodeMultiDraw(pass, indirectBuf(200), {
    indexed: true,
    useNative: true,
    count: 6,
    visibility: [1, 1, 0, 1, 1, 1],
  });
  assert.deepEqual(pass.calls, [
    ['multiDrawIndexedIndirect', 0, 2],
    ['multiDrawIndexedIndirect', 60, 3],
  ]);
  assert.deepEqual(r, { calls: 2, draws: 5, skipped: 1, native: true });

  pass.calls.length = 0;
  encodeMultiDraw(pass, indirectBuf(200), { useNative: true, count: 4 });
  assert.deepEqual(pass.calls, [['multiDrawIndirect', 0, 4]]);
});

test('useNative falls back to emulation when the pass lacks the method', () => {
  const pass = createMockPass(false);
  const r = encodeMultiDraw(pass, indirectBuf(100), { useNative: true, count: 2 });
  assert.equal(r.native, false);
  assert.deepEqual(pass.calls, [['drawIndirect', 0], ['drawIndirect', 16]]);
});

test('IndirectArgsBuffer.encodeDraws covers handed-out slots by default', () => {
  const { device } = createMockDevice();
  const args = new IndirectArgsBuffer({ device, capacity: 4 });
  args.allocate(rec(1));
  args.allocate(rec(2));
  const pass = createMockPass();
  const r = args.encodeDraws(pass, { visibility: [true, false] });
  assert.deepEqual(pass.calls, [['drawIndexedIndirect', 0]]);
  assert.equal(r.skipped, 1);
});
