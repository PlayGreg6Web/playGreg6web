// Test-only harness: runs lifted wasm64 functions on several worker threads that share one
// `WebAssembly.Memory` (memory64, shared). The module must be linked with --import-memory and
// --shared-memory.
// Usage: node [--experimental-wasm-memory64] runner_threads.js <module.wasm> '<spec JSON>'
//
// spec: { pages, gprOffset, guestBase, init: [[addr, hex]], read: [[addr, len]],
//              workers: [{ state, stackTop, shadowStack, calls: [{ fn, regs: [[index, value]] }] }] }
// `init`/`read` addresses and `stackTop` are guest VAs (linear address = VA + guestBase, wrapping);
// `state` and `shadowStack` are linear (host) addresses.
// Every worker instantiates the module on the shared memory, then runs its calls in order (each
// with a fresh register set and a pushed return address); all workers meet at a barrier before
// every call so that the same call runs on all threads at once. The barrier spins briefly before
// it sleeps, so on a machine with enough cores the workers leave it within nanoseconds.
// Prints the `read` ranges as hex, comma separated (or `error: ...`), then one line per call
// index: `overlap <i> <n>`, where n is the largest number of workers any worker saw inside call i
// (itself included) when it entered the call. Overlap is measured by a shared counter that the
// workers bump around the call, not by wall-clock timestamps, so it cannot be fooled by coarse
// clocks or by one worker finishing before a late one starts: n >= 2 means two calls really were
// in flight at once.
'use strict';
const fs = require('fs');
const { Worker, isMainThread, workerData, parentPort } = require('worker_threads');

const RETURN_ADDRESS = 0xC0DEC0DEn;
const RSP = 4;
const lin = (guestBase, addr) => Number(BigInt.asUintN(64, BigInt(addr) + BigInt(guestBase || 0)));

// Control block (Int32Array): [generation, arrived, active, unused, peak of call 0, peak of call 1, ...]
const GENERATION = 0;
const ARRIVED = 1;
const ACTIVE = 2;
const PEAKS = 4;
const SPIN_MS = 20;

// Reusable barrier for `n` threads. Waiters spin first (all threads leave together when enough
// CPUs are free) and fall back to Atomics.wait when the machine is oversubscribed.
function barrier(ctrl, n) {
  const generation = Atomics.load(ctrl, GENERATION);
  if (Atomics.add(ctrl, ARRIVED, 1) === n - 1) {
    Atomics.store(ctrl, ARRIVED, 0);
    Atomics.add(ctrl, GENERATION, 1);
    Atomics.notify(ctrl, GENERATION);
  } else {
    const until = performance.now() + SPIN_MS;
    while (Atomics.load(ctrl, GENERATION) === generation && performance.now() < until) {
      // spin
    }
    while (Atomics.load(ctrl, GENERATION) === generation) Atomics.wait(ctrl, GENERATION, generation, 5);
  }
}

function recordPeak(ctrl, call, seen) {
  for (;;) {
    const peak = Atomics.load(ctrl, PEAKS + call);
    if (peak >= seen || Atomics.compareExchange(ctrl, PEAKS + call, peak, seen) === peak) return;
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
    const base = lin(spec.guestBase, addr);
    for (let i = 0; i < hex.length; i += 2) bytes[base + i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  const control = new Int32Array(new SharedArrayBuffer(4 * (PEAKS + spec.workers[0].calls.length)));
  let pending = spec.workers.length;
  let failure = null;
  const count = spec.workers.length;
  const workers = spec.workers.map((w) => new Worker(__filename, { workerData: { module, memory, control, count, w, gprOffset: spec.gprOffset, guestBase: spec.guestBase } }));
  const finish = () => {
    pending -= 1;
    if (pending !== 0) return;
    if (failure) {
      process.stdout.write(`error: ${failure}\n`);
      return;
    }
    const out = spec.read.map(([addr, len]) => {
      const base = lin(spec.guestBase, addr);
      let s = '';
      for (let i = 0; i < len; i++) s += bytes[base + i].toString(16).padStart(2, '0');
      return s;
    });
    process.stdout.write(out.join(',') + '\n');
    for (let i = 0; i < spec.workers[0].calls.length; i++) {
      process.stdout.write(`overlap ${i} ${Atomics.load(control, PEAKS + i)}\n`);
    }
  };
  for (const worker of workers) {
    worker.on('message', (msg) => {
      if (msg !== 'done') failure = failure || msg;
      finish();
    });
    worker.on('error', (e) => {
      failure = failure || String(e);
      finish();
    });
  }
} else {
  const { module, memory, control, count, w, gprOffset, guestBase } = workerData;
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
  let error = setupError;
  for (let index = 0; index < w.calls.length; index++) {
    const call = w.calls[index];
    if (!error) {
      try {
        prepare(call);
      } catch (e) {
        error = e;
      }
    }
    barrier(control, count);
    if (error) continue;
    try {
      // Count this call in flight; seeing another one in flight proves real overlap.
      recordPeak(control, index, Atomics.add(control, ACTIVE, 1) + 1);
      try {
        instance.exports[call.fn](BigInt(w.state));
      } finally {
        Atomics.sub(control, ACTIVE, 1);
      }
    } catch (e) {
      error = e;
    }
  }
  parentPort.postMessage(error ? String(error && error.message ? error.message : error) : 'done');

  function prepare(call) {
    const view = new DataView(memory.buffer);
    const state = BigInt(w.state);
    const gpr = Number(state) + gprOffset;
    for (let i = 0; i < 16; i++) view.setBigUint64(gpr + i * 8, 0n, true);
    for (const [i, v] of call.regs) view.setBigUint64(gpr + i * 8, BigInt(v), true);
    const rsp = BigInt(w.stackTop) - 8n;
    view.setBigUint64(lin(guestBase, rsp), RETURN_ADDRESS, true);
    view.setBigUint64(gpr + RSP * 8, rsp, true);
  }
}
