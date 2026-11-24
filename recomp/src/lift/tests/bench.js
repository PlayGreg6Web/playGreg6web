// R19 micro-benchmarks: times lifted functions under node (memory64).
// Usage: node bench.js <module.wasm> '<spec JSON>'
// spec: { state, gprOffset, guestBase, memory (as in runner.js, optional), rsp, samples,
//         benches: [{ name, fn, kind: "array"|"chase"|"copy"|"structs", base, n, dst, calls }] }
// Prints one line per bench: name \t median ns/element \t min \t max \t checksum.
'use strict';
const fs = require('fs');

const [wasmPath, specJson] = process.argv.slice(2);
const spec = JSON.parse(specJson);
const STATE = Number(BigInt(spec.state));
const GPR = STATE + spec.gprOffset;
const GUEST_BASE = BigInt(spec.guestBase || 0);
const MEM = spec.memory || null;
const TRAP_DELTA = 0x8000000000000000n;
let instance;
let tableAddr = 0;
const view = () => new DataView(instance.exports.memory.buffer);
const index = (a) => {
  if (!MEM) return -1;
  if (MEM.model === 'window') {
    const rel = BigInt.asUintN(64, a - BigInt(MEM.base));
    return rel < BigInt(MEM.size) ? Number(rel >> 16n) : -1;
  }
  const idx = a >> 16n;
  return Number(idx < BigInt(MEM.pages) ? idx : BigInt(MEM.pages));
};
const lin = (va) => {
  const a = BigInt.asUintN(64, BigInt(va));
  const i = index(a);
  const d = i < 0 ? GUEST_BASE : view().getBigUint64(tableAddr + i * 8, true);
  return Number(BigInt.asUintN(64, a + d));
};
const set64 = (va, x) => {
  // Writes byte by byte so that a qword never straddles a remapped slot wrongly.
  for (let i = 0n; i < 8n; i++) view().setUint8(lin(BigInt(va) + i), Number((BigInt(x) >> (8n * i)) & 0xFFn));
};
const SCRATCH = 0x7FF000;
const byte = (va) => view().getUint8(lin(va));
const put = (va, b) => view().setUint8(lin(va), b);
const imports = {
  env: {
    __playgta6_indirect_call() { throw new Error('indirect call'); },
    __playgta6_unimplemented(s, va) { throw new Error(`unimplemented at 0x${va.toString(16)}`); },
    __playgta6_divide_error() { throw new Error('divide error'); },
    __playgta6_trap(s, va) { throw new Error(`trap at 0x${va.toString(16)}`); },
    __playgta6_guest_read(va, n) {
      for (let i = 0n; i < n; i++) view().setUint8(SCRATCH + Number(i), byte(va + i));
      return BigInt(SCRATCH);
    },
    __playgta6_guest_scratch: () => BigInt(SCRATCH),
    __playgta6_guest_write(va, n) {
      for (let i = 0n; i < n; i++) put(va + i, view().getUint8(SCRATCH + Number(i)));
    },
    __playgta6_guest_memmove(dst, src, n) {
      // Segment-wise like the runtime: contiguous host runs inside one slot are copied at once.
      let done = 0n;
      const tmp = new Uint8Array(Number(n));
      while (done < n) {
        const a = src + done;
        const run = BigInt(Math.min(Number(n - done), 0x10000 - Number(a & 0xFFFFn)));
        tmp.set(new Uint8Array(instance.exports.memory.buffer, lin(a), Number(run)), Number(done));
        done += run;
      }
      done = 0n;
      while (done < n) {
        const a = dst + done;
        const run = BigInt(Math.min(Number(n - done), 0x10000 - Number(a & 0xFFFFn)));
        new Uint8Array(instance.exports.memory.buffer, lin(a), Number(run)).set(tmp.subarray(Number(done), Number(done + run)));
        done += run;
      }
    },
    __playgta6_guest_memset(dst, value, n) {
      for (let i = 0n; i < n; i++) put(dst + i, Number(value) & 0xFF);
    },
  },
};
instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), imports);
if (MEM) {
  const sym = instance.exports[MEM.model === 'window' ? '__playgta6_window_table' : '__playgta6_page_table'];
  tableAddr = Number(sym.value);
  const entries = MEM.model === 'window' ? Number(BigInt(MEM.size) >> 16n) : MEM.pages + 1;
  for (let i = 0; i < entries; i++) {
    const d = MEM.model === 'window' || i === MEM.pages ? TRAP_DELTA : GUEST_BASE;
    view().setBigUint64(tableAddr + i * 8, d, true);
  }
  for (const [va, linear] of MEM.slots) {
    const a = BigInt(va) & ~0xFFFFn;
    view().setBigUint64(tableAddr + index(a) * 8, linear === null ? TRAP_DELTA : BigInt.asUintN(64, BigInt(linear) - a), true);
  }
}

function setup(b) {
  const base = BigInt(b.base);
  if (b.kind === 'array' || b.kind === 'copy') {
    for (let i = 0; i < b.n; i++) set64(base + BigInt(i) * 8n, BigInt(i) * 3n + 1n);
  } else if (b.kind === 'chase') {
    const perm = Array.from({ length: b.n }, (_, i) => i);
    let seed = 12345;
    for (let i = b.n - 1; i > 0; i--) {
      seed = (seed * 1103515245 + 12345) >>> 0;
      const j = seed % (i + 1);
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    for (let i = 0; i < b.n; i++) {
      set64(base + BigInt(perm[i]) * 64n, base + BigInt(perm[(i + 1) % b.n]) * 64n);
    }
  } else if (b.kind === 'structs') {
    for (let i = 0; i < b.n; i++) {
      const s = base + BigInt(i) * 32n;
      set64(s, BigInt(i));
      set64(s + 8n, BigInt(i) * 7n);
      set64(s + 16n, 3n);
      set64(s + 24n, 0n);
    }
  }
}

function call(b) {
  const v = view();
  for (let i = 0; i < 16; i++) v.setBigUint64(GPR + i * 8, 0n, true);
  const steps = b.kind === 'chase' ? b.steps : b.n;
  v.setBigUint64(GPR + 1 * 8, BigInt(steps), true);  // rcx
  v.setBigUint64(GPR + 2 * 8, BigInt(b.base), true);  // rdx
  if (b.dst) v.setBigUint64(GPR + 8 * 8, BigInt(b.dst), true);  // r8
  const rsp = BigInt(spec.rsp) - 8n;
  set64(rsp, 0xC0DEC0DEn);
  v.setBigUint64(GPR + 4 * 8, rsp, true);
  instance.exports[b.fn](BigInt(STATE));
  return view().getBigUint64(GPR, true);
}

const out = [];
for (const b of spec.benches) {
  setup(b);
  let checksum = call(b);  // warm-up
  for (let i = 0; i < 2; i++) checksum = call(b);
  const per = [];
  const elements = (b.kind === 'chase' ? b.steps : b.n) * b.calls;
  for (let s = 0; s < spec.samples; s++) {
    const t0 = process.hrtime.bigint();
    for (let c = 0; c < b.calls; c++) checksum = call(b);
    const t1 = process.hrtime.bigint();
    per.push(Number(t1 - t0) / elements);
  }
  per.sort((x, y) => x - y);
  out.push(`${b.name}\t${per[per.length >> 1].toFixed(3)}\t${per[0].toFixed(3)}\t${per[per.length - 1].toFixed(3)}\t${checksum.toString(16)}`);
}
process.stdout.write(out.join('\n') + '\n');
