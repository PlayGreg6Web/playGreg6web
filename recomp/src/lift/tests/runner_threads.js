// Test-only harness: runs lifted wasm64 functions on several worker threads that share one
// `WebAssembly.Memory` (memory64, shared). The module must be linked with --import-memory and
// --shared-memory.
// Usage: node [--experimental-wasm-memory64] runner_threads.js <module.wasm> '<spec JSON>'
//
// spec: { pages, gprOffset, init: [[addr, hex]], read: [[addr, len]],
//              workers: [{ state, stackTop, shadowStack, calls: [{ fn, regs: [[index, value]] }] }] }
// Every worker instantiates the module on the shared memory, then runs its calls in order (each
// with a fresh register set and a pushed return address); all workers meet at a barrier before
// every call so that the same call runs on all threads at once.
// Prints the `read` ranges as hex, comma separated (or `error: ...`), then one line per call
// index: `overlap <i> <0|1>`, 1 when every worker was inside call i at the same time.
'use strict';
const fs = require('fs');
const { Worker, isMainThread, workerData, parentPort } = require('worker_threads');

const RETURN_ADDRESS = 0xC0DEC0DEn;
const RSP = 4;

// Reusable barrier for `n` threads on an Int32Array: [generation, arrived].
function barrier(ctrl, n) {
  const generation = Atomics.load(ctrl, 0);
  if (Atomics.add(ctrl, 1, 1) === n - 1) {
    Atomics.store(ctrl, 1, 0);
    Atomics.add(ctrl, 0, 1);
    Atomics.notify(ctrl, 0);
  } else {
    while (Atomics.load(ctrl, 0) === generation) Atomics.wait(ctrl, 0, generation);
  }
}

function imports(memory) {
  const fail = (what) => (state, va) => {
    throw new Error(`${what} at 0x${va.toString(16)}`);
  };
  return {
    env: {
      memory,
      __playgta6_indirect_call: fail('indirect call'),
      __playgta6_unimplemented: fail('unimplemented instruction'),
      __playgta6_divide_error: fail('divide error'),
      __playgta6_trap: fail('trap'),
    },
  };
}

if (isMainThread) {
  const [wasmPath, specJson] = process.argv.slice(2);
  const spec = JSON.parse(specJson);
  const module = new WebAssembly.Module(fs.readFileSync(wasmPath));
  // The memory64 descriptor key changed across V8 versions; use the first one that links.
  let memory = null;
  let lastError = null;
  for (const key of ['address', 'index']) {
    for (const size of [spec.pages, BigInt(spec.pages)]) {
      try {
        const m = new WebAssembly.Memory({ initial: size, maximum: size, shared: true, [key]: 'i64' });
        new WebAssembly.Instance(module, imports(m));
        memory = m;
        break;
      } catch (e) {
        lastError = e;
      }
    }
    if (memory) break;
  }
  if (!memory) {
    // Older engines cannot create a memory64 from JavaScript: a capability, not a failure.
    process.stdout.write(`skip: this node cannot create a shared memory64 from JavaScript (${lastError})\n`);
    process.exit(0);
  }
  const bytes = new Uint8Array(memory.buffer);
  for (const [addr, hex] of spec.init) {
    const base = Number(BigInt(addr));
    for (let i = 0; i < hex.length; i += 2) bytes[base + i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  const control = new Int32Array(new SharedArrayBuffer(8));
  let pending = spec.workers.length;
  let failure = null;
  const count = spec.workers.length;
  const workers = spec.workers.map((w) => new Worker(__filename, { workerData: { module, memory, control, count, w, gprOffset: spec.gprOffset } }));
  const times = [];
  const finish = () => {
    pending -= 1;
    if (pending !== 0) return;
    if (failure) {
      process.stdout.write(`error: ${failure}\n`);
      return;
    }
    const out = spec.read.map(([addr, len]) => {
      const base = Number(BigInt(addr));
      let s = '';
      for (let i = 0; i < len; i++) s += bytes[base + i].toString(16).padStart(2, '0');
      return s;
    });
    process.stdout.write(out.join(',') + '\n');
    for (let i = 0; i < times[0].length; i++) {
      const start = Math.max(...times.map((t) => t[i][0]));
      const end = Math.min(...times.map((t) => t[i][1]));
      process.stdout.write(`overlap ${i} ${start < end ? 1 : 0}\n`);
    }
  };
  for (const worker of workers) {
    worker.on('message', (msg) => {
      if (msg && msg.times) times.push(msg.times);
      else failure = failure || msg;
      finish();
    });
    worker.on('error', (e) => {
      failure = failure || String(e);
      finish();
    });
  }
} else {
  const { module, memory, control, count, w, gprOffset } = workerData;
  let instance = null;
  let setupError = null;
  try {
    instance = new WebAssembly.Instance(module, imports(memory));
    // Each thread gets its own LLVM shadow stack (if the module uses one at all).
    const sp = instance.exports.__stack_pointer;
    if (sp instanceof WebAssembly.Global) sp.value = BigInt(w.shadowStack);
  } catch (e) {
    setupError = e;
  }
  // A worker that failed to set up still takes part in every barrier.
  const times = [];
  let error = setupError;
  for (const call of w.calls) {
    barrier(control, count);
    if (error) continue;
    try {
      run(call);
    } catch (e) {
      error = e;
    }
  }
  parentPort.postMessage(error ? String(error && error.message ? error.message : error) : { times });

  function run(call) {
    const view = new DataView(memory.buffer);
    const state = BigInt(w.state);
    const gpr = Number(state) + gprOffset;
    for (let i = 0; i < 16; i++) view.setBigUint64(gpr + i * 8, 0n, true);
    for (const [i, v] of call.regs) view.setBigUint64(gpr + i * 8, BigInt(v), true);
    const rsp = BigInt(w.stackTop) - 8n;
    view.setBigUint64(Number(rsp), RETURN_ADDRESS, true);
    view.setBigUint64(gpr + RSP * 8, rsp, true);
    const start = performance.timeOrigin + performance.now();
    instance.exports[call.fn](state);
    times.push([start, performance.timeOrigin + performance.now()]);
  }
}
