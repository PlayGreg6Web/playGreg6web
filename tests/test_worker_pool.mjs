import test from 'node:test';
import assert from 'node:assert/strict';

import {
  WorkerPool,
  PRIORITY,
  WorkerPoolError,
  PoolTerminatedError,
  QueueFullError,
  JobAbortedError,
  JobTimeoutError,
  WorkerCrashError,
  CircuitOpenError,
  serveWorkerPool,
  withTransfer,
  serializeError,
  deserializeError,
  defaultPoolSize,
} from '../web/workers/worker_pool.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));

/** Manual clock providing injectable now/setTimeout/clearTimeout. */
function createClock() {
  let time = 0;
  let nextId = 1;
  const timers = new Map();
  return {
    now: () => time,
    setTimeout(fn, ms) {
      const id = nextId++;
      timers.set(id, { at: time + ms, fn });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    advance(ms) {
      const target = time + ms;
      for (;;) {
        let best = null;
        for (const [id, t] of timers) {
          if (t.at <= target && (!best || t.at < best.t.at)) best = { id, t };
        }
        if (!best) break;
        timers.delete(best.id);
        time = Math.max(time, best.t.at);
        best.t.fn();
      }
      time = target;
    },
    get pending() {
      return timers.size;
    },
  };
}

/** Worker double with scriptable replies, errors and termination. */
class FakeWorker {
  constructor() {
    this.listeners = { message: [], error: [], messageerror: [], exit: [] };
    this.received = [];
    this.transfers = [];
    this.terminated = false;
    this.autoReply = null;
  }
  addEventListener(type, fn) {
    this.listeners[type].push(fn);
  }
  postMessage(msg, transfer = []) {
    if (this.terminated) throw new Error('postMessage on terminated worker');
    this.received.push(msg);
    this.transfers.push(transfer);
    if (this.autoReply && msg.type) {
      queueMicrotask(() => this.autoReply(msg, this));
    }
  }
  terminate() {
    this.terminated = true;
  }
  emit(type, ev) {
    if (this.terminated) return;
    for (const fn of this.listeners[type]) fn(ev);
  }
  reply(msg) {
    this.emit('message', { data: msg });
  }
  jobs() {
    return this.received.filter((m) => m.type);
  }
  cancels() {
    return this.received.filter((m) => m.cancel);
  }
  last() {
    return this.jobs().at(-1);
  }
}

function setup(options = {}) {
  const clock = createClock();
  const workers = [];
  const pool = new WorkerPool({
    workerFactory: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w;
    },
    maxWorkers: 2,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    ...options,
  });
  return { pool, workers, clock };
}

function settleState(promise) {
  const state = { status: 'pending', value: undefined, count: 0 };
  promise.then(
    (v) => {
      state.status = 'resolved';
      state.value = v;
      state.count++;
    },
    (e) => {
      state.status = 'rejected';
      state.value = e;
      state.count++;
    }
  );
  return state;
}

test('constructor validates options', () => {
  assert.throws(() => new WorkerPool({}), TypeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxWorkers: 0 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxWorkers: 2, minWorkers: 3 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), overflow: 'drop' }), RangeError);
});

test('defaultPoolSize clamps and reserves headroom', () => {
  assert.equal(defaultPoolSize({ hardwareConcurrency: 16 }), 8);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 6 }), 4);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 1 }), 1);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 16, reserve: 4, max: 6 }), 6);
  assert.equal(defaultPoolSize({ hardwareConcurrency: NaN }), 2);
});

test('lazy spawn, bounded concurrency and result delivery', async () => {
  const { pool, workers, clock } = setup();
  assert.equal(workers.length, 0);
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  const c = pool.submit('t', 3);
  assert.equal(workers.length, 2);
  assert.equal(pool.stats().busy, 2);
  assert.equal(pool.stats().queued, 1);
  assert.deepEqual(workers[0].last(), { id: 1, type: 't', payload: 1 });

  workers[0].reply({ id: 1, ok: true, result: 'A' });
  assert.equal(await a, 'A');
  // third job moved onto the freed worker, never exceeding two workers
  assert.equal(workers.length, 2);
  assert.equal(workers[0].last().payload, 3);
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  workers[0].reply({ id: 3, ok: true, result: 'C' });
  assert.deepEqual(await Promise.all([b, c]), ['B', 'C']);
  const s = pool.stats();
  assert.equal(s.completed, 3);
  assert.equal(s.workers, 2);
  assert.equal(s.busy, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('idle workers are shrunk down to minWorkers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 3, minWorkers: 1, idleTimeoutMs: 100 });
  const jobs = [pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3)];
  workers.forEach((w, i) => w.reply({ id: i + 1, ok: true, result: i }));
  await Promise.all(jobs);
  assert.equal(pool.stats().workers, 3);
  clock.advance(99);
  assert.equal(pool.stats().workers, 3);
  clock.advance(1);
  assert.equal(pool.stats().workers, 1);
  assert.equal(workers.filter((w) => w.terminated).length, 2);
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('a busy worker is not shrunk and its idle timer is cancelled', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, idleTimeoutMs: 100 });
  const a = pool.submit('t', 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  clock.advance(50);
  const b = pool.submit('t', 2);
  clock.advance(500);
  assert.equal(workers[0].terminated, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await b;
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('FIFO within a priority level and priority ordering across levels', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const blocker = pool.submit('t', 'blocker');
  const order = [];
  const track = (label, p) => p.then(() => order.push(label));
  const all = [
    track('low1', pool.submit('t', 'low1', { priority: 'low' })),
    track('norm1', pool.submit('t', 'norm1')),
    track('high1', pool.submit('t', 'high1', { priority: PRIORITY.HIGH })),
    track('norm2', pool.submit('t', 'norm2', { priority: 'normal' })),
    track('high2', pool.submit('t', 'high2', { priority: 'high' })),
    track('low2', pool.submit('t', 'low2', { priority: 'low' })),
  ];
  const started = [];
  workers[0].reply({ id: workers[0].last().id, ok: true, result: null });
  await blocker;
  for (let i = 0; i < 6; i++) {
    started.push(workers[0].last().payload);
    workers[0].reply({ id: workers[0].last().id, ok: true, result: null });
    await flush();
  }
  await Promise.all(all);
  assert.deepEqual(started, ['high1', 'high2', 'norm1', 'norm2', 'low1', 'low2']);
  assert.deepEqual(order, started);
  await pool.terminate();
});

test('invalid priority and type reject without throwing synchronously', async () => {
  const { pool } = setup();
  await assert.rejects(pool.submit('t', 1, { priority: 'urgent' }), RangeError);
  await assert.rejects(pool.submit('', 1), TypeError);
  assert.equal(pool.stats().submitted, 0);
  await pool.terminate();
});

test('transfer list is forwarded to postMessage', async () => {
  const { pool, workers } = setup();
  const buf = new ArrayBuffer(16);
  const p = pool.submit('inflate', { buf }, { transfer: [buf] });
  assert.equal(workers[0].transfers[0][0], buf);
  workers[0].reply({ id: 1, ok: true, result: 'x' });
  await p;
  await pool.terminate();
});

test('real postMessage transfer detaches the caller buffer (structured clone)', async () => {
  const { pool, workers } = setup();
  const buf = new ArrayBuffer(32);
  const p = pool.submit('t', buf, { transfer: [buf] });
  // The fake does not detach; emulate what a real postMessage does with structuredClone.
  const clone = structuredClone(buf, { transfer: workers[0].transfers[0] });
  assert.equal(buf.byteLength, 0);
  assert.equal(clone.byteLength, 32);
  workers[0].reply({ id: 1, ok: true, result: clone });
  assert.equal(await p, clone);
  await pool.terminate();
});

test('postMessage failure rejects the job only; worker stays usable', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const warm = pool.submit('t', 0);
  workers[0].reply({ id: 1, ok: true, result: 0 });
  await warm;
  const orig = workers[0].postMessage.bind(workers[0]);
  workers[0].postMessage = () => {
    throw new DOMException('cannot clone', 'DataCloneError');
  };
  await assert.rejects(pool.submit('t', () => {}), /cannot clone/);
  workers[0].postMessage = orig;
  const ok = pool.submit('t', 2);
  workers[0].reply({ id: 3, ok: true, result: 'fine' });
  assert.equal(await ok, 'fine');
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('queue limit with reject policy', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  await assert.rejects(pool.submit('t', 'c'), (e) => e instanceof QueueFullError && e.limit === 1);
  assert.equal(pool.stats().queued, 1);
  assert.equal(pool.stats().rejected, 1);
  workers[0].reply({ id: 1, ok: true, result: 'A' });
  await a;
  workers[0].reply({ id: 2, ok: true, result: 'B' });
  await b;
  await pool.terminate();
});

test('queue limit with wait policy applies backpressure in order', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'wait' });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  const c = pool.submit('t', 'c');
  const d = pool.submit('t', 'd');
  assert.equal(pool.stats().queued, 1);
  assert.equal(pool.stats().blocked, 2);
  const seen = [];
  for (let i = 1; i <= 4; i++) {
    seen.push(workers[0].last().payload);
    workers[0].reply({ id: i, ok: true, result: i });
    await flush();
  }
  assert.deepEqual(await Promise.all([a, b, c, d]), [1, 2, 3, 4]);
  assert.deepEqual(seen, ['a', 'b', 'c', 'd']);
  await pool.terminate();
});

test('maxQueue 0 only accepts work that can start immediately', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 0 });
  const a = pool.submit('t', 'a');
  await assert.rejects(pool.submit('t', 'b'), QueueFullError);
  workers[0].reply({ id: 1, ok: true, result: 'A' });
  await a;
  await pool.terminate();
});

test('aborting a blocked job removes it', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 0, overflow: 'wait' });
  const a = pool.submit('t', 'a');
  const ac = new AbortController();
  const b = pool.submit('t', 'b', { signal: ac.signal });
  assert.equal(pool.stats().blocked, 1);
  ac.abort();
  await assert.rejects(b, JobAbortedError);
  assert.equal(pool.stats().blocked, 0);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  await pool.terminate();
});

test('per-job timeout rejects, requests cancel, then hard-kills after grace', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 50 });
  const p = pool.submit('slow', 1, { timeoutMs: 100 });
  const state = settleState(p);
  clock.advance(99);
  await flush();
  assert.equal(state.status, 'pending');
  clock.advance(1);
  await flush();
  assert.equal(state.status, 'rejected');
  assert.ok(state.value instanceof JobTimeoutError);
  assert.equal(state.value.timeoutMs, 100);
  assert.deepEqual(workers[0].cancels(), [{ id: 1, cancel: true }]);
  clock.advance(50);
  assert.equal(workers[0].terminated, true);
  assert.equal(pool.stats().hardKills, 1);
  assert.equal(pool.stats().workers, 0);
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('defaultTimeoutMs applies and timeout timer is cleared on success', async () => {
  const { pool, workers, clock } = setup({ defaultTimeoutMs: 500 });
  const p = pool.submit('t', 1);
  assert.equal(clock.pending, 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await p;
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('abort while queued rejects immediately and never reaches a worker', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const ac = new AbortController();
  const b = pool.submit('t', 'b', { signal: ac.signal });
  ac.abort(new Error('nope'));
  await assert.rejects(b, (e) => e.name === 'AbortError' && e instanceof JobAbortedError && e.reason.message === 'nope');
  assert.equal(pool.stats().queued, 0);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  assert.equal(workers[0].jobs().length, 1);
  assert.equal(pool.stats().cancelled, 1);
  await pool.terminate();
});

test('already-aborted signal rejects at submit', async () => {
  const { pool, workers } = setup();
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(pool.submit('t', 1, { signal: ac.signal }), JobAbortedError);
  assert.equal(workers.length, 0);
  await pool.terminate();
});

test('abort while running: cooperative ack keeps the worker alive', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 100 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  const queued = pool.submit('t', 2);
  const state = settleState(p);
  ac.abort();
  await flush();
  assert.equal(state.status, 'rejected');
  assert.equal(state.count, 1);
  assert.deepEqual(workers[0].cancels(), [{ id: 1, cancel: true }]);
  // worker is occupied until it acknowledges; queued job waits
  assert.equal(workers[0].jobs().length, 1);
  workers[0].reply({ id: 1, cancelled: true });
  assert.equal(workers[0].jobs().length, 2);
  assert.equal(workers[0].terminated, false);
  clock.advance(1000); // grace timer must have been cleared
  assert.equal(workers[0].terminated, false);
  workers[0].reply({ id: 2, ok: true, result: 'q' });
  assert.equal(await queued, 'q');
  assert.equal(state.count, 1);
  assert.equal(pool.stats().hardKills, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('abort while running: a late result also counts as acknowledgement and is dropped', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  workers[0].reply({ id: 1, ok: true, result: 'too late' });
  assert.equal(pool.stats().busy, 0);
  assert.equal(pool.stats().completed, 0);
  await pool.terminate();
});

test('abort while running: no ack within grace hard-kills and replaces the worker', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 100 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  const next = pool.submit('t', 2);
  const state = settleState(p);
  ac.abort();
  await flush();
  assert.equal(state.status, 'rejected');
  clock.advance(100);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 2);
  workers[1].reply({ id: 2, ok: true, result: 'next' });
  assert.equal(await next, 'next');
  assert.equal(state.count, 1);
  assert.equal(pool.stats().hardKills, 1);
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('cancelGraceMs 0 kills the worker immediately', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, cancelGraceMs: 0 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(workers[0].terminated, true);
  await pool.terminate();
});

test('worker error mid-job rejects with WorkerCrashError, replaces worker, queue continues', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  workers[0].emit('error', { message: 'boom' });
  await assert.rejects(a, (e) => e instanceof WorkerCrashError && e.reason === 'boom' && e instanceof WorkerPoolError);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 'b');
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  assert.equal(await b, 'B');
  const s = pool.stats();
  assert.equal(s.crashed, 1);
  assert.equal(s.failed, 1);
  assert.equal(s.completed, 1);
  await pool.terminate();
});

test('messageerror and exit events are treated as crashes', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  workers[0].emit('messageerror', {});
  await assert.rejects(a, (e) => e instanceof WorkerCrashError && /messageerror/.test(e.reason));
  const b = pool.submit('t', 'b');
  workers[1].emit('exit', { code: 1 });
  await assert.rejects(b, (e) => e instanceof WorkerCrashError && /exited/.test(e.reason));
  assert.equal(pool.stats().crashed, 2);
  await pool.terminate();
});

test('idle worker crash is replaced lazily and not attributed to any job', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  workers[0].emit('error', { message: 'idle boom' });
  assert.equal(pool.stats().workers, 0);
  const b = pool.submit('t', 'b');
  assert.equal(workers.length, 2);
  workers[1].reply({ id: 2, ok: true, result: 2 });
  assert.equal(await b, 2);
  await pool.terminate();
});

test('retry policy retries crashes but never job-level errors', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, retries: 2, breaker: false });
  const p = pool.submit('t', 'x');
  workers[0].emit('error', { message: 'c1' });
  assert.equal(workers.length, 2);
  workers[1].emit('error', { message: 'c2' });
  assert.equal(workers.length, 3);
  workers[2].reply({ id: 1, ok: true, result: 'third time' });
  assert.equal(await p, 'third time');
  assert.equal(pool.stats().retried, 2);

  const q = pool.submit('t', 'y');
  workers[2].emit('error', { message: 'c3' });
  workers[3].emit('error', { message: 'c4' });
  workers[4].emit('error', { message: 'c5' });
  await assert.rejects(q, WorkerCrashError);

  const r = pool.submit('t', 'z');
  const spawned = workers.length;
  const w = workers.at(-1);
  w.reply({ id: w.last().id, ok: false, error: { name: 'Error', message: 'job failed' } });
  await assert.rejects(r, /job failed/);
  assert.equal(workers.length, spawned); // a job-level error neither retries nor respawns
  assert.equal(pool.stats().retried, 4);
  await pool.terminate();
});

test('jobs with a transfer list are not retried after a crash', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, retries: 3 });
  const buf = new ArrayBuffer(8);
  const p = pool.submit('t', buf, { transfer: [buf] });
  workers[0].emit('error', { message: 'boom' });
  await assert.rejects(p, WorkerCrashError);
  assert.equal(pool.stats().retried, 0);
  await pool.terminate();
});

test('circuit breaker opens after K crashes in the window and stops respawning', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, breaker: { maxCrashes: 3, windowMs: 1000 } });
  const jobs = [pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3), pool.submit('t', 4)];
  const settled = jobs.map(settleState);
  workers[0].emit('error', { message: 'a' });
  workers[1].emit('error', { message: 'b' });
  assert.equal(pool.stats().circuitOpen, false);
  workers[2].emit('error', { message: 'c' });
  await flush();
  assert.equal(pool.stats().circuitOpen, true);
  assert.equal(workers.length, 3); // no respawn
  assert.ok(settled[0].value instanceof WorkerCrashError);
  assert.ok(settled[1].value instanceof WorkerCrashError);
  assert.ok(settled[2].value instanceof WorkerCrashError);
  // queued job 4 was rejected by the breaker, not left hanging
  assert.ok(settled[3].value instanceof CircuitOpenError);
  assert.equal(settled[3].value.crashes, 3);
  await assert.rejects(pool.submit('t', 5), CircuitOpenError);
  assert.equal(workers.length, 3);
  pool.resetCircuit();
  const ok = pool.submit('t', 6);
  assert.equal(workers.length, 4);
  workers[3].reply({ id: 5, ok: true, result: 'recovered' });
  assert.equal(await ok, 'recovered');
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('crashes outside the breaker window do not accumulate', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, breaker: { maxCrashes: 2, windowMs: 100 } });
  const a = pool.submit('t', 1);
  workers[0].emit('error', { message: 'a' });
  await assert.rejects(a, WorkerCrashError);
  clock.advance(200);
  const b = pool.submit('t', 2);
  workers[1].emit('error', { message: 'b' });
  await assert.rejects(b, WorkerCrashError);
  assert.equal(pool.stats().circuitOpen, false);
  await pool.terminate();
});

test('worker factory failure cannot spin forever', async () => {
  let calls = 0;
  const pool = new WorkerPool({
    workerFactory: () => {
      calls++;
      throw new Error('script missing');
    },
    maxWorkers: 2,
    breaker: { maxCrashes: 3, windowMs: 1000 },
  });
  const results = await Promise.allSettled([pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3), pool.submit('t', 4)]);
  assert.ok(calls <= 4);
  assert.ok(results.every((r) => r.status === 'rejected'));
  assert.ok(results[0].reason instanceof WorkerCrashError);
  assert.match(results[0].reason.reason, /script missing/);
  await pool.terminate();
  // and with the breaker disabled a failing factory still terminates the pump loop
  const pool2 = new WorkerPool({
    workerFactory: () => {
      throw new Error('nope');
    },
    breaker: false,
  });
  await assert.rejects(Promise.all([pool2.submit('t', 1), pool2.submit('t', 2)]), WorkerCrashError);
  await pool2.terminate();
});

test('error envelope round-trips into a real Error', async () => {
  const { pool, workers } = setup();
  const p = pool.submit('t', 1);
  const err = new RangeError('bad range');
  err.code = 'E_RANGE';
  workers[0].reply({ id: 1, ok: false, error: serializeError(err) });
  await assert.rejects(p, (e) => {
    assert.ok(e instanceof RangeError);
    assert.equal(e.message, 'bad range');
    assert.equal(e.code, 'E_RANGE');
    assert.equal(e.remote, true);
    assert.equal(e.stack, err.stack);
    return true;
  });
  const custom = deserializeError({ name: 'InflateError', message: 'corrupt', stack: 'S' });
  assert.ok(custom instanceof Error);
  assert.equal(custom.name, 'InflateError');
  assert.deepEqual(serializeError('str'), { name: 'Error', message: 'str' });
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('progress messages reach onProgress; a throwing callback is contained', async () => {
  const { pool, workers } = setup();
  const seen = [];
  const p = pool.submit('t', 1, {
    onProgress: (v) => {
      seen.push(v);
      throw new Error('callback bug');
    },
  });
  workers[0].reply({ id: 1, progress: 0.25 });
  workers[0].reply({ id: 1, progress: 0.5 });
  workers[0].reply({ id: 1, ok: true, result: 'done' });
  workers[0].reply({ id: 1, progress: 1 }); // after settle: ignored
  assert.equal(await p, 'done');
  assert.deepEqual(seen, [0.25, 0.5]);
  await pool.terminate();
});

test('unknown, late, duplicate and malformed messages are ignored', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const p = pool.submit('t', 1);
  workers[0].reply({ id: 999, ok: true, result: 'stray' });
  workers[0].reply(null);
  workers[0].reply('junk');
  workers[0].reply({ ok: true, result: 'no id' });
  assert.equal(pool.stats().completed, 0);
  workers[0].reply({ id: 1, ok: true, result: 'real' });
  workers[0].reply({ id: 1, ok: true, result: 'duplicate' });
  workers[0].reply({ id: 1, ok: false, error: { message: 'dup err' } });
  assert.equal(await p, 'real');
  const s = pool.stats();
  assert.equal(s.completed, 1);
  assert.equal(s.failed, 0);
  await pool.terminate();
});

test('events from a replaced worker are ignored', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  const old = workers[0];
  old.emit('error', { message: 'x' });
  await assert.rejects(a, WorkerCrashError);
  // terminated worker's emit is a no-op in the fake; invoke listeners directly to prove the pool guards
  for (const fn of old.listeners.message) fn({ data: { id: 2, ok: true, result: 'stale' } });
  for (const fn of old.listeners.error) fn({ message: 'again' });
  assert.equal(pool.stats().crashed, 1);
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  assert.equal(await b, 'B');
  await pool.terminate();
});

test('drain resolves when idle', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  await pool.drain();
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  let drained = false;
  const d = pool.drain().then(() => {
    drained = true;
  });
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  assert.equal(drained, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await d;
  await Promise.all([a, b]);
  assert.equal(drained, true);
  await pool.terminate();
});

test('graceful terminate finishes outstanding work, then rejects new submits', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  let done = false;
  const t = pool.terminate({ graceful: true }).then(() => {
    done = true;
  });
  await assert.rejects(pool.submit('t', 3), PoolTerminatedError);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  assert.equal(done, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await t;
  assert.deepEqual(await Promise.all([a, b]), [1, 2]);
  assert.equal(workers[0].terminated, true);
  assert.equal(pool.stats().workers, 0);
  assert.equal(clock.pending, 0);
});

test('graceful terminate on an idle pool resolves immediately', async () => {
  const { pool, workers } = setup();
  const a = pool.submit('t', 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  await pool.terminate({ graceful: true });
  assert.equal(workers[0].terminated, true);
});

test('immediate terminate rejects queued and running jobs and kills workers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'wait', defaultTimeoutMs: 1000 });
  const ac = new AbortController();
  const running = settleState(pool.submit('t', 1, { signal: ac.signal }));
  const queued = settleState(pool.submit('t', 2));
  const blocked = settleState(pool.submit('t', 3));
  const dr = pool.drain();
  await pool.terminate();
  await dr;
  await flush();
  for (const s of [running, queued, blocked]) {
    assert.equal(s.status, 'rejected');
    assert.ok(s.value instanceof PoolTerminatedError);
    assert.equal(s.count, 1);
  }
  assert.equal(workers[0].terminated, true);
  assert.equal(clock.pending, 0);
  ac.abort(); // listener was removed; must be a no-op
  await assert.rejects(pool.submit('t', 4), PoolTerminatedError);
});

test('terminate is idempotent and can upgrade a graceful shutdown', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = settleState(pool.submit('t', 1));
  const t1 = pool.terminate({ graceful: true });
  const t2 = pool.terminate();
  assert.equal(t1, t2);
  await t2;
  await flush();
  assert.equal(a.status, 'rejected');
  assert.equal(workers[0].terminated, true);
  await pool.terminate();
  await pool.terminate({ graceful: true });
});

test('terminate cancels a worker that was mid hard-cancel without leaking timers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 500 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(clock.pending, 1);
  await pool.terminate();
  assert.equal(workers[0].terminated, true);
  assert.equal(clock.pending, 0);
});

test('stats report average wait and run time from the injected clock', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  clock.advance(10);
  workers[0].reply({ id: 1, ok: true, result: 1 }); // a: wait 0, run 10; b starts: wait 10
  await a;
  clock.advance(30);
  workers[0].reply({ id: 2, ok: true, result: 2 }); // b: run 30
  await b;
  const s = pool.stats();
  assert.equal(s.avgWaitMs, 5);
  assert.equal(s.avgRunMs, 20);
  assert.equal(s.submitted, 2);
  assert.equal(s.spawned, 1);
  await pool.terminate();
});

test('every job settles exactly once across mixed outcomes', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 2, retries: 1, cancelGraceMs: 10 });
  const ac = new AbortController();
  const states = [
    settleState(pool.submit('t', 'ok')),
    settleState(pool.submit('t', 'abort', { signal: ac.signal })),
    settleState(pool.submit('t', 'queued-abort', { signal: ac.signal })),
    settleState(pool.submit('t', 'tmo', { timeoutMs: 5 })),
  ];
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  ac.abort();
  clock.advance(5);
  clock.advance(100);
  workers[0].reply({ id: 1, ok: true, result: 'dup' });
  workers[1].reply({ id: 2, cancelled: true });
  await pool.terminate();
  await flush();
  for (const s of states) assert.equal(s.count, 1);
  assert.equal(clock.pending, 0);
});

// ---- end-to-end against serveWorkerPool through in-memory ports -----------

/** Build a pair of connected message scopes with microtask delivery. */
function createPortPair() {
  const make = () => ({ listeners: new Set(), peer: null, closed: false });
  const a = make();
  const b = make();
  a.peer = b;
  b.peer = a;
  const wrap = (side) => ({
    addEventListener(type, fn) {
      if (type === 'message') side.listeners.add(fn);
    },
    removeEventListener(type, fn) {
      side.listeners.delete(fn);
    },
    postMessage(msg, transfer = []) {
      if (side.closed) return;
      const data = structuredClone(msg, { transfer });
      setImmediate(() => {
        if (side.peer.closed) return;
        for (const fn of [...side.peer.listeners]) fn({ data });
      });
    },
  });
  return { pool: wrap(a), worker: wrap(b), sideA: a, sideB: b };
}

/** FakeWorker-like adapter whose far end is a real serveWorkerPool scope. */
function createLoopbackFactory(handlers, created = []) {
  return () => {
    const { pool: poolPort, worker: workerPort, sideB } = createPortPair();
    serveWorkerPool(handlers, { scope: workerPort });
    const listeners = { message: [], error: [], messageerror: [], exit: [] };
    poolPort.addEventListener('message', (ev) => listeners.message.forEach((fn) => fn(ev)));
    const w = {
      terminated: false,
      addEventListener(type, fn) {
        listeners[type].push(fn);
      },
      postMessage: (msg, transfer) => poolPort.postMessage(msg, transfer),
      terminate() {
        w.terminated = true;
        sideB.closed = true;
      },
    };
    created.push(w);
    return w;
  };
}

test('e2e: serveWorkerPool handles jobs, transfers, progress and errors', async () => {
  const handlers = {
    sum: async (payload) => payload.reduce((a, b) => a + b, 0),
    inflate: async (payload, ctx) => {
      const out = new Uint8Array(payload.buf.byteLength * 2);
      ctx.progress(0.5);
      ctx.transfer(out.buffer);
      return { size: out.length, buf: out.buffer };
    },
    wrapped: async () => withTransfer({ tag: 'w' }, []),
    plain: async () => ({ result: 'unwrapped', transfer: [] }),
    boom: async () => {
      const e = new TypeError('bad input');
      e.code = 'E_BAD';
      throw e;
    },
  };
  const created = [];
  const pool = new WorkerPool({ workerFactory: createLoopbackFactory(handlers, created), maxWorkers: 2 });
  assert.equal(await pool.submit('sum', [1, 2, 3]), 6);

  const input = new Uint8Array([1, 2, 3, 4]).buffer;
  const progress = [];
  const res = await pool.submit('inflate', { buf: input }, { transfer: [input], onProgress: (v) => progress.push(v) });
  assert.equal(input.byteLength, 0); // detached by transfer
  assert.equal(res.size, 8);
  assert.equal(res.buf.byteLength, 8);
  assert.deepEqual(progress, [0.5]);

  assert.deepEqual(await pool.submit('wrapped', null), { tag: 'w' });
  assert.equal(await pool.submit('plain', null), 'unwrapped');

  await assert.rejects(pool.submit('boom', null), (e) => e instanceof TypeError && e.code === 'E_BAD' && e.remote === true);
  await assert.rejects(pool.submit('nope', null), (e) => e instanceof TypeError && /Unknown job type/.test(e.message));
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('e2e: cooperative cancellation through ctx.signal keeps the worker', async () => {
  let observed = false;
  const handlers = {
    wait: (_p, ctx) =>
      new Promise((resolve) => {
        ctx.signal.addEventListener('abort', () => {
          observed = true;
          resolve('ignored');
        });
      }),
    quick: async () => 'quick',
  };
  const created = [];
  const pool = new WorkerPool({
    workerFactory: createLoopbackFactory(handlers, created),
    maxWorkers: 1,
    cancelGraceMs: 10000,
  });
  const ac = new AbortController();
  const p = pool.submit('wait', null, { signal: ac.signal });
  const q = pool.submit('quick', null);
  await flush();
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(await q, 'quick');
  assert.equal(observed, true);
  assert.equal(created.length, 1);
  assert.equal(created[0].terminated, false);
  assert.equal(pool.stats().hardKills, 0);
  await pool.terminate();
});

test('e2e: a handler that ignores cancellation is hard-killed after the grace period', async () => {
  const clock = createClock();
  const handlers = { hang: () => new Promise(() => {}), quick: async () => 'quick' };
  const created = [];
  const pool = new WorkerPool({
    workerFactory: createLoopbackFactory(handlers, created),
    maxWorkers: 1,
    cancelGraceMs: 50,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
  });
  const p = pool.submit('hang', null, { timeoutMs: 20 });
  const q = pool.submit('quick', null);
  await flush();
  clock.advance(20);
  await assert.rejects(p, JobTimeoutError);
  await flush();
  clock.advance(50);
  assert.equal(created[0].terminated, true);
  assert.equal(await q, 'quick');
  assert.equal(created.length, 2);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('serveWorkerPool ignores duplicate ids, malformed messages and unknown cancels; unsubscribe works', async () => {
  const sent = [];
  const listeners = new Set();
  const scope = {
    addEventListener: (t, fn) => listeners.add(fn),
    removeEventListener: (t, fn) => listeners.delete(fn),
    postMessage: (m) => sent.push(m),
  };
  let runs = 0;
  let release;
  const stop = serveWorkerPool(
    {
      slow: () => {
        runs++;
        return new Promise((r) => (release = r));
      },
    },
    { scope }
  );
  const send = (data) => listeners.forEach((fn) => fn({ data }));
  send(null);
  send({ type: 'slow' }); // no id
  send({ id: 1, type: 'slow', payload: 0 });
  send({ id: 1, type: 'slow', payload: 0 }); // duplicate while active
  send({ id: 77, cancel: true }); // unknown id: acked, harmless
  assert.deepEqual(sent, [{ id: 77, cancelled: true }]);
  release('v');
  await flush();
  assert.equal(runs, 1);
  assert.deepEqual(sent.at(-1), { id: 1, ok: true, result: 'v' });
  stop();
  assert.equal(listeners.size, 0);
});

test('serveWorkerPool reports a result that cannot be cloned as a job error', async () => {
  const sent = [];
  const listeners = new Set();
  const scope = {
    addEventListener: (t, fn) => listeners.add(fn),
    postMessage: (m) => {
      if (m.ok === true) throw new DOMException('no clone', 'DataCloneError');
      sent.push(m);
    },
  };
  serveWorkerPool({ bad: async () => () => {} }, { scope });
  listeners.forEach((fn) => fn({ data: { id: 5, type: 'bad', payload: null } }));
  await flush();
  assert.equal(sent[0].id, 5);
  assert.equal(sent[0].ok, false);
  assert.equal(sent[0].error.name, 'DataCloneError');
});
