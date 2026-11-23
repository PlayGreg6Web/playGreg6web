// Test-only harness: runs lifted wasm64 functions under node (memory64).
// Usage: node [--experimental-wasm-memory64] runner.js <module.wasm> <cases.json>
//
// cases.json: { state, gprOffset, flagsOffset, xmmOffset, cases: [{ fn, regs, flags[6], mem, read, out }] }
// `regs` lists [index, value] pairs; unlisted registers are zero.
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
const RSP = 4;
const RETURN_ADDRESS = 0xC0DEC0DEn;

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

const lines = spec.cases.map((c, index) => {
  let v = view();
  for (let i = 0; i < 16; i++) v.setBigUint64(GPR + i * 8, 0n, true);
  for (const [i, r] of c.regs) v.setBigUint64(GPR + i * 8, BigInt(r), true);
  const regs0 = new Map(c.regs.map(([i, r]) => [i, BigInt(r)]));
  c.flags.forEach((f, i) => v.setUint8(FLAGS + i, f));
  for (const [addr, bytes] of c.mem) {
    const base = Number(BigInt(addr));
    for (let i = 0; i < bytes.length; i += 2) v.setUint8(base + i / 2, parseInt(bytes.substr(i, 2), 16));
  }
  const rsp = (regs0.get(RSP) || 0n) - 8n;
  v.setBigUint64(Number(rsp), RETURN_ADDRESS, true);
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
    const base = Number(BigInt(addr));
    let s = '';
    for (let i = 0; i < len; i++) s += v.getUint8(base + i).toString(16).padStart(2, '0');
    return s;
  }).join(',');
  return `${index}\t${regs}\t${flags}\t${mem}\t${error}`;
});
process.stdout.write(lines.join('\n') + '\n');
