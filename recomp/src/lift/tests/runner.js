// Test-only harness: runs lifted wasm64 functions under node (memory64).
// Usage: node [--experimental-wasm-memory64] runner.js <module.wasm> <cases.json>
//
// cases.json: { state, stateSize, guestBase, gprOffset, flagsOffset, xmmOffset, mxcsrOffset, resumeRipOffset,
//               unwindPendingOffset, cases: [{ fn, regs, flags[6], mem, read, out }] }
// `regs` lists [index, value] pairs; unlisted registers are zero.
// `mem`/`read` addresses and rsp are guest VAs: guest VA v lives at linear address
// v + guestBase (64-bit wrapping), the translation the module was linked with. Addresses inside
// the state block [state, state + stateSize) are host addresses and are used as they are.
// Optional `memory` (R19 table models, lift/memory.rs): { model: "window", base, size, slots } or
// { model: "page", pages, slots }; `slots` lists [slot guest VA, linear address or null =
// unmapped]. The harness fills the exported delta table like the runtime would (every other slot:
// the guestBase translation for "page", unmapped for "window"), translates every guest address the
// same way and provides the __playgta6_guest_* slow-path helpers.
// For every case the harness writes guest registers/flags/memory, pushes a fake return address
// (emulating the caller's `call`), invokes the export and prints one tab-separated line:
//   <index> \t <regs listed in `out`, hex, comma separated> \t <CF PF AF ZF SF OF> \t <memory reads, hex> \t <error>
'use strict';
const fs = require('fs');

const [wasmPath, casesPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
const STATE = Number(BigInt(spec.state));
const GPR = STATE + spec.gprOffset;
const FLAGS = STATE + spec.flagsOffset;
const XMM = STATE + (spec.xmmOffset || 0);
// MXCSR of a new guest thread (the runtime initialises it like this).
const MXCSR = spec.mxcsrOffset ? STATE + spec.mxcsrOffset : 0;
const RESUME_RIP = spec.resumeRipOffset ? STATE + spec.resumeRipOffset : 0;
const UNWIND_PENDING = spec.unwindPendingOffset ? STATE + spec.unwindPendingOffset : 0;
const RSP = 4;
const RETURN_ADDRESS = 0xC0DEC0DEn;
const GUEST_BASE = BigInt(spec.guestBase || 0);
const STATE_END = STATE + (spec.stateSize || 0);
const MEM = spec.memory || null;
const TRAP_DELTA = 0x8000000000000000n;  // unmapped slot: the address lands beyond any memory
const SCRATCH = 0x7FF000;  // linear scratch buffer of the slow-path helpers
let tableAddr = 0;
// Table index of guest VA `a` (BigInt) under the table model, or -1 for the flat translation.
const tableIndex = (a) => {
  if (!MEM) return -1;
  if (MEM.model === 'window') {
    const rel = BigInt.asUintN(64, a - BigInt(MEM.base));
    return rel < BigInt(MEM.size) ? Number(rel >> 16n) : -1;
  }
  const idx = a >> 16n;
  return Number(idx < BigInt(MEM.pages) ? idx : BigInt(MEM.pages));
};
const delta = (a) => {
  const i = tableIndex(a);
  return i < 0 ? GUEST_BASE : view().getBigUint64(tableAddr + i * 8, true);
};
// Guest VA -> linear address (the state block is host memory).
const lin = (addr) => {
  const a = BigInt.asUintN(64, BigInt(addr));
  if (a >= BigInt(STATE) && a < BigInt(STATE_END)) return Number(a);
  const h = BigInt.asUintN(64, a + delta(a));
  if (h >= TRAP_DELTA) throw new Error(`unmapped guest page at ${hex(a)}`);
  return Number(h);
};
const getByte = (va) => view().getUint8(lin(va));
const setByte = (va, b) => view().setUint8(lin(va), b);
const rd64 = (va) => {
  let x = 0n;
  for (let i = 7; i >= 0; i--) x = (x << 8n) | BigInt(getByte(BigInt(va) + BigInt(i)));
  return x;
};
// Maps the slot containing guest VA `va` at linear address `linear` (null: unmapped).
const mapSlot = (va, linear) => {
  const a = BigInt.asUintN(64, BigInt(va)) & ~0xFFFFn;
  const i = tableIndex(a);
  if (i < 0) throw new Error(`no table slot for ${hex(a)}`);
  const d = linear === null ? TRAP_DELTA : BigInt.asUintN(64, BigInt(linear) - a);
  view().setBigUint64(tableAddr + i * 8, d, true);
};

// Host-implemented "imports" reachable through __playgta6_indirect_call, keyed by guest VA.
// 0x7FFE0000: rax = rcx * 10 + rdx, then `ret` (pops the return address like a real callee).
const hostFunctions = new Map([
  [0x7FFE0000n, () => {
    const v = view();
    const rcx = v.getBigUint64(GPR + 8, true);
    const rdx = v.getBigUint64(GPR + 16, true);
    v.setBigUint64(GPR, BigInt.asUintN(64, rcx * 10n + rdx), true);
    v.setBigUint64(GPR + RSP * 8, v.getBigUint64(GPR + RSP * 8, true) + 8n, true);
  }],
  // 0x7FFE0010: double f(double x = xmm0, double y = xmm1, int64 n = r8) = x * y + n, result in
  // xmm0 (upper half preserved, like the import thunks), then `ret`.
  [0x7FFE0010n, () => {
    const v = view();
    const x = v.getFloat64(XMM, true);
    const y = v.getFloat64(XMM + 16, true);
    const n = v.getBigInt64(GPR + 8 * 8, true);
    v.setFloat64(XMM, x * y + Number(n), true);
    v.setBigUint64(GPR + RSP * 8, v.getBigUint64(GPR + RSP * 8, true) + 8n, true);
  }],
  // 0x7FFE0020: stands in for RtlRestoreContext. rcx points at { rip, mask, gpr[16] }: every GPR
  // whose bit is set in mask is restored, then the restore is left pending at rip (resume
  // protocol: no `ret`, the lifted frames unwind until the owner resumes).
  [0x7FFE0020n, () => {
    const v = view();
    const desc = v.getBigUint64(GPR + 8, true);
    const rip = rd64(desc);
    const mask = Number(rd64(desc + 8n));
    for (let i = 0; i < 16; i++) {
      if (mask & (1 << i)) v.setBigUint64(GPR + i * 8, rd64(desc + 16n + BigInt(i) * 8n), true);
    }
    v.setBigUint64(RESUME_RIP, rip, true);
    v.setUint8(UNWIND_PENDING, 1);
  }],
  // 0x7FFE0030: remaps the table slot holding guest VA rcx to linear address rdx (0 = unmap), the
  // way XMemMapPhysicalPagesScatter changes the window under running code; then `ret`.
  [0x7FFE0030n, () => {
    const v = view();
    const va = v.getBigUint64(GPR + 8, true);
    const linear = v.getBigUint64(GPR + 16, true);
    mapSlot(va, linear === 0n ? null : linear);
    v.setBigUint64(GPR + RSP * 8, v.getBigUint64(GPR + RSP * 8, true) + 8n, true);
  }],
]);

let instance;
const lifted = new Map();
const view = () => new DataView(instance.exports.memory.buffer);
const hex = (v) => '0x' + v.toString(16);

const imports = {
  env: {
    __playgta6_indirect_call(state, target) {
      const fn = lifted.get(target) || hostFunctions.get(target);
      if (!fn) throw new Error(`indirect call to unknown target ${hex(target)}`);
      fn(state);
    },
    __playgta6_unimplemented(state, va) {
      throw new Error(`unimplemented instruction at ${hex(va)}`);
    },
    __playgta6_divide_error(state, va) {
      throw new Error(`divide error at ${hex(va)}`);
    },
    __playgta6_trap(state, va) {
      throw new Error(`trap at ${hex(va)}`);
    },
    // Deterministic stand-ins for the runtime-chosen instructions: they read and write the
    // guest GPRs in the state block like the real runtime helpers (32-bit results zero-extended).
    __playgta6_cpuid(state) {
      const leaf = getReg32(state, 0);
      const subleaf = getReg32(state, 1);
      setRegs(state, [[0, leaf ^ 0x11111111], [3, subleaf ^ 0x22222222], [1, 0x33333333], [2, 0x44444444]]);
    },
    __playgta6_rdtsc(state) {
      tsc += 0x1_0000_0001n;
      setRegs(state, [[0, Number(tsc & 0xFFFFFFFFn)], [2, Number(tsc >> 32n)]]);
    },
    __playgta6_rdtscp(state) {
      tsc += 0x1_0000_0001n;
      setRegs(state, [[0, Number(tsc & 0xFFFFFFFFn)], [2, Number(tsc >> 32n)], [1, 7]]);
    },
    __playgta6_xgetbv(state) {
      if (getReg32(state, 1) !== 0) throw new Error('xgetbv: #GP');
      setRegs(state, [[0, 7], [2, 0]]);
    },
    // R19 slow paths: byte by byte through the same translation (an unmapped slot throws).
    __playgta6_guest_read(va, n) {
      for (let i = 0n; i < n; i++) view().setUint8(SCRATCH + Number(i), getByte(va + i));
      return BigInt(SCRATCH);
    },
    __playgta6_guest_scratch() {
      return BigInt(SCRATCH);
    },
    __playgta6_guest_write(va, n) {
      for (let i = 0n; i < n; i++) setByte(va + i, view().getUint8(SCRATCH + Number(i)));
    },
    __playgta6_guest_memmove(dst, src, n) {
      const tmp = [];
      for (let i = 0n; i < n; i++) tmp.push(getByte(src + i));
      tmp.forEach((b, i) => setByte(dst + BigInt(i), b));
    },
    __playgta6_guest_memset(dst, value, n) {
      for (let i = 0n; i < n; i++) setByte(dst + i, Number(value) & 0xFF);
    },
  },
};

let tsc = 0x1_0000_0000n;
const gprAddr = (state, i) => Number(state) + spec.gprOffset + i * 8;
const getReg32 = (state, i) => view().getUint32(gprAddr(state, i), true);
const setRegs = (state, pairs) => {
  const v = view();
  for (const [i, value] of pairs) v.setBigUint64(gprAddr(state, i), BigInt(value >>> 0), true);
};

instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), imports);
for (const [name, fn] of Object.entries(instance.exports)) {
  const m = /^sub_([0-9A-F]+)$/.exec(name);
  if (m) lifted.set(BigInt('0x' + m[1]), fn);
}
if (MEM) {
  const sym = instance.exports[MEM.model === 'window' ? '__playgta6_window_table' : '__playgta6_page_table'];
  if (sym) {
    tableAddr = Number(sym.value);
    const entries = MEM.model === 'window' ? Number(BigInt(MEM.size) >> 16n) : MEM.pages + 1;
    const v = view();
    for (let i = 0; i < entries; i++) {
      const d = MEM.model === 'window' || i === MEM.pages ? TRAP_DELTA : GUEST_BASE;
      v.setBigUint64(tableAddr + i * 8, d, true);
    }
    for (const [va, linear] of MEM.slots) mapSlot(BigInt(va), linear === null ? null : BigInt(linear));
  }
}

const lines = spec.cases.map((c, index) => {
  let v = view();
  for (let i = 0; i < 16; i++) v.setBigUint64(GPR + i * 8, 0n, true);
  for (const [i, r] of c.regs) v.setBigUint64(GPR + i * 8, BigInt(r), true);
  const regs0 = new Map(c.regs.map(([i, r]) => [i, BigInt(r)]));
  c.flags.forEach((f, i) => v.setUint8(FLAGS + i, f));
  if (MXCSR) v.setUint32(MXCSR, 0x1F80, true);
  if (UNWIND_PENDING) v.setUint8(UNWIND_PENDING, 0);
  for (const [addr, bytes] of c.mem) {
    const base = BigInt(addr);
    for (let i = 0; i < bytes.length; i += 2) setByte(base + BigInt(i / 2), parseInt(bytes.substr(i, 2), 16));
  }
  const rsp = (regs0.get(RSP) || 0n) - 8n;
  for (let i = 0n; i < 8n; i++) setByte(rsp + i, Number((RETURN_ADDRESS >> (8n * i)) & 0xFFn));
  v.setBigUint64(GPR + RSP * 8, rsp, true);

  let error = '';
  try {
    instance.exports[c.fn](BigInt(STATE));
  } catch (e) {
    error = String(e && e.message ? e.message : e).replace(/\s+/g, ' ');
  }

  v = view();
  const regs = c.out.map((i) => hex(v.getBigUint64(GPR + i * 8, true))).join(',');
  let flags = '';
  for (let i = 0; i < 6; i++) flags += v.getUint8(FLAGS + i);
  const mem = c.read.map(([addr, len]) => {
    const base = BigInt(addr);
    let s = '';
    for (let i = 0; i < len; i++) s += getByte(base + BigInt(i)).toString(16).padStart(2, '0');
    return s;
  }).join(',');
  return `${index}\t${regs}\t${flags}\t${mem}\t${error}`;
});
process.stdout.write(lines.join('\n') + '\n');
