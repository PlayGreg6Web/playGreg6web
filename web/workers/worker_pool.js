// Bounded Web Worker pool and offscreen task dispatcher.
//
// Intended for background decompression and shader pre-parsing: jobs carry
// transferable ArrayBuffers, can be prioritised, timed out and cancelled, and a
// crashing worker script is contained by replacement, retry and a circuit breaker.
//
// Pure ES module, no top-level side effects. Works on the main thread and inside
// a worker, and under Node with an injected worker factory (see tests/test_worker_pool.mjs).
//
// ---------------------------------------------------------------------------
// Message protocol (structured-clone friendly plain objects)
//
//   pool -> worker
//     { id, type, payload }        run job `id` with handler `type`
//     { id, cancel: true }         cooperative cancel request for job `id`
//
//   worker -> pool
//     { id, ok: true,  result }    job finished
//     { id, ok: false, error }     job threw; error = { name, message, stack, code? }
//     { id, progress }             optional progress value (any cloneable)
//     { id, cancelled: true }      acknowledgement of a cancel request
//
//   Messages with an unknown, finished or duplicate id are ignored.
//
// Ownership rule for transferables
//   Buffers listed in `submit(..., { transfer })` are TRANSFERRED to the worker and
//   are detached (byteLength 0) in the caller as soon as `submit` returns. Never
//   reuse them. Results may come back with transferables (see `serveWorkerPool`);
//   the receiving side then owns them. Because a transferred buffer cannot be sent
//   twice, a job submitted with a non-empty transfer list is never retried after a
//   worker crash.
// ---------------------------------------------------------------------------

/** Job priorities. Lower value runs first; FIFO within one level. */
export const PRIORITY = Object.freeze({ HIGH: 0, NORMAL: 1, LOW: 2 });

const PRIORITY_NAMES = Object.freeze({ high: 0, normal: 1, low: 2 });
const PRIORITY_LEVELS = 3;

export const DEFAULT_CANCEL_GRACE_MS = 1000;
export const DEFAULT_BREAKER_MAX_CRASHES = 5;
export const DEFAULT_BREAKER_WINDOW_MS = 10000;

/** Base class for all errors raised by the pool itself. */
export class WorkerPoolError extends Error {
  constructor(message, code, options) {
    super(message, options);
    this.name = 'WorkerPoolError';
    this.code = code;
  }
}

/** The pool was terminated (or is terminating) and cannot accept or finish the job. */
export class PoolTerminatedError extends WorkerPoolError {
  constructor(message = 'Worker pool terminated') {
    super(message, 'POOL_TERMINATED');
    this.name = 'PoolTerminatedError';
  }
}

/** The bounded queue is full and the overflow policy is `'reject'`. */
export class QueueFullError extends WorkerPoolError {
  constructor(limit) {
    super(`Worker pool queue is full (limit ${limit})`, 'QUEUE_FULL');
    this.name = 'QueueFullError';
    this.limit = limit;
  }
}

/** The job was cancelled through its AbortSignal. Named `AbortError` like DOM aborts. */
export class JobAbortedError extends WorkerPoolError {
  constructor(reason) {
    super('Worker job aborted', 'ABORTED', { cause: reason });
    this.name = 'AbortError';
    this.reason = reason;
  }
}

/** The job exceeded its per-job timeout. */
export class JobTimeoutError extends WorkerPoolError {
  constructor(timeoutMs) {
    super(`Worker job timed out after ${timeoutMs} ms`, 'TIMEOUT');
    this.name = 'JobTimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/** The worker running the job crashed (error, messageerror, exit or spawn failure). */
export class WorkerCrashError extends WorkerPoolError {
  constructor(reason, cause) {
    super(`Worker crashed: ${reason}`, 'WORKER_CRASH', cause === undefined ? undefined : { cause });
    this.name = 'WorkerCrashError';
    this.reason = reason;
  }
}

/** Too many crashes within the breaker window; the pool refuses further work. */
export class CircuitOpenError extends WorkerPoolError {
  constructor(crashes, windowMs, cause) {
    super(
      `Worker pool circuit open: ${crashes} worker crashes within ${windowMs} ms`,
      'CIRCUIT_OPEN',
      cause === undefined ? undefined : { cause }
    );
    this.name = 'CircuitOpenError';
    this.crashes = crashes;
    this.windowMs = windowMs;
  }
}

const STANDARD_ERRORS = {
  Error,
  TypeError,
  RangeError,
  SyntaxError,
  ReferenceError,
  EvalError,
  URIError,
};

/**
 * Serialise any thrown value into a structured-clone safe envelope.
 * @param {unknown} err
 * @returns {{ name: string, message: string, stack?: string, code?: string|number }}
 */
export function serializeError(err) {
  if (err !== null && typeof err === 'object') {
    const out = {
      name: typeof err.name === 'string' ? err.name : 'Error',
      message: typeof err.message === 'string' ? err.message : String(err.message ?? ''),
    };
    if (typeof err.stack === 'string') out.stack = err.stack;
    if (typeof err.code === 'string' || typeof err.code === 'number') out.code = err.code;
    return out;
  }
  return { name: 'Error', message: String(err) };
}

/**
 * Rebuild a real `Error` from a serialised envelope. The result is flagged
 * `remote = true` so callers can tell worker-side failures from pool errors.
 * @param {{ name?: string, message?: string, stack?: string, code?: string|number }} data
 * @returns {Error}
 */
export function deserializeError(data) {
  const src = data !== null && typeof data === 'object' ? data : { message: String(data) };
  const name = typeof src.name === 'string' ? src.name : 'Error';
  const Ctor = Object.prototype.hasOwnProperty.call(STANDARD_ERRORS, name)
    ? STANDARD_ERRORS[name]
    : Error;
  const err = new Ctor(typeof src.message === 'string' ? src.message : '');
  err.name = name;
  if (typeof src.stack === 'string') err.stack = src.stack;
  if (src.code !== undefined) err.code = src.code;
  err.remote = true;
  return err;
}

/**
 * Suggested pool size: hardware threads minus headroom for the game's own
 * pthreads, clamped to `[min, max]`.
 * @param {{ hardwareConcurrency?: number, reserve?: number, min?: number, max?: number }} [opts]
 * @returns {number}
 */
export function defaultPoolSize(opts = {}) {
  const { reserve = 2, min = 1, max = 8 } = opts;
  let hc = opts.hardwareConcurrency;
  if (hc === undefined) {
    hc = typeof navigator !== 'undefined' && navigator ? navigator.hardwareConcurrency : undefined;
  }
  if (!Number.isFinite(hc) || hc < 1) hc = 4;
  return Math.max(min, Math.min(max, Math.floor(hc) - reserve));
}

function normalisePriority(priority) {
  if (priority === undefined || priority === null) return PRIORITY.NORMAL;
  if (typeof priority === 'string' && Object.prototype.hasOwnProperty.call(PRIORITY_NAMES, priority)) {
    return PRIORITY_NAMES[priority];
  }
  if (Number.isInteger(priority) && priority >= 0 && priority < PRIORITY_LEVELS) return priority;
  throw new RangeError(`Invalid priority: ${String(priority)}`);
}

/** Attach a listener to either a DOM-style or a Node-style worker. */
function listen(target, event, handler) {
  if (typeof target.addEventListener === 'function') {
    target.addEventListener(event, handler);
  } else if (typeof target.on === 'function') {
    // Node worker_threads: message/error/exit hand over the raw value.
    target.on(event, (value) => {
      if (event === 'message') handler({ data: value });
      else if (event === 'error') handler({ error: value, message: value && value.message });
      else handler({ code: value });
    });
  }
}

function describeErrorEvent(ev) {
  if (!ev) return 'unknown error';
  const msg = ev.message || (ev.error && ev.error.message);
  return msg ? String(msg) : 'unknown error';
}

/**
 * Bounded pool of dedicated workers speaking the protocol above.
 */
export class WorkerPool {
  /**
   * @param {object} options
   * @param {() => object} [options.workerFactory] Returns a Worker-like object.
   * @param {string|URL} [options.url] Used when no factory is given: `new Worker(url, workerOptions)`.
   * @param {object} [options.workerOptions] Default `{ type: 'module' }`.
   * @param {number} [options.maxWorkers] Default `defaultPoolSize()`.
   * @param {number} [options.minWorkers=0] Idle shrink never goes below this.
   * @param {number} [options.idleTimeoutMs=0] Terminate surplus idle workers after this long (0 = never).
   * @param {number} [options.maxQueue=Infinity] Maximum queued (not yet running) jobs.
   * @param {'reject'|'wait'} [options.overflow='reject'] Policy when the queue is full.
   * @param {number} [options.cancelGraceMs=1000] Grace before a cancelled running job's worker is terminated.
   * @param {number} [options.retries=0] Retries per job after a worker crash (never after a job error).
   * @param {number} [options.defaultTimeoutMs=0] Per-job run timeout default (0 = none).
   * @param {{ maxCrashes?: number, windowMs?: number }|false} [options.breaker] Circuit breaker; `false` disables.
   * @param {() => number} [options.now] Clock in ms.
   * @param {Function} [options.setTimeout]
   * @param {Function} [options.clearTimeout]
   */
  constructor(options = {}) {
    const {
      workerFactory,
      url,
      workerOptions = { type: 'module' },
      maxWorkers = defaultPoolSize(),
      minWorkers = 0,
      idleTimeoutMs = 0,
      maxQueue = Infinity,
      overflow = 'reject',
      cancelGraceMs = DEFAULT_CANCEL_GRACE_MS,
      retries = 0,
      defaultTimeoutMs = 0,
      breaker = {},
      now,
      setTimeout: setTimeoutFn,
      clearTimeout: clearTimeoutFn,
    } = options;

    if (typeof workerFactory !== 'function' && url === undefined) {
      throw new TypeError('WorkerPool requires a workerFactory function or a url');
    }
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1) {
      throw new RangeError(`maxWorkers must be a positive integer, got ${maxWorkers}`);
    }
    if (!Number.isInteger(minWorkers) || minWorkers < 0 || minWorkers > maxWorkers) {
      throw new RangeError(`minWorkers must be an integer in [0, maxWorkers], got ${minWorkers}`);
    }
    if (overflow !== 'reject' && overflow !== 'wait') {
      throw new RangeError(`overflow must be 'reject' or 'wait', got ${overflow}`);
    }
    if (!(maxQueue >= 0)) throw new RangeError(`maxQueue must be >= 0, got ${maxQueue}`);
    if (!Number.isInteger(retries) || retries < 0) {
      throw new RangeError(`retries must be a non-negative integer, got ${retries}`);
    }

    this._factory =
      typeof workerFactory === 'function' ? workerFactory : () => new Worker(url, workerOptions);
    this._maxWorkers = maxWorkers;
    this._minWorkers = minWorkers;
    this._idleTimeoutMs = idleTimeoutMs;
    this._maxQueue = maxQueue;
    this._overflow = overflow;
    this._cancelGraceMs = cancelGraceMs;
    this._retries = retries;
    this._defaultTimeoutMs = defaultTimeoutMs;
    this._breakerMax = breaker === false ? 0 : (breaker.maxCrashes ?? DEFAULT_BREAKER_MAX_CRASHES);
    this._breakerWindow = breaker === false ? 0 : (breaker.windowMs ?? DEFAULT_BREAKER_WINDOW_MS);
    this._now =
      now ||
      (() => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()));
    this._setTimeout = setTimeoutFn || ((fn, ms) => globalThis.setTimeout(fn, ms));
    this._clearTimeout = clearTimeoutFn || ((t) => globalThis.clearTimeout(t));

    this._slots = new Set();
    this._queues = [[], [], []];
    this._blocked = [];
    this._running = 0;
    this._nextId = 1;
    this._crashTimes = [];
    this._lastCrash = null;
    this._circuitOpen = false;
    this._closed = false;
    this._terminatePromise = null;
    this._terminateState = null;
    this._drainWaiters = [];

    this._counters = {
      submitted: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      rejected: 0,
      crashed: 0,
      retried: 0,
      hardKills: 0,
      spawned: 0,
    };
    this._waitTotal = 0;
    this._waitCount = 0;
    this._runTotal = 0;
    this._runCount = 0;
  }

  /**
   * Submit a job.
   * @param {string} type Handler name registered with `serveWorkerPool` in the worker.
   * @param {*} payload Structured-cloneable job input.
   * @param {object} [opts]
   * @param {Transferable[]} [opts.transfer] Transferred (detached in the caller) with the payload.
   * @param {'high'|'normal'|'low'|number} [opts.priority='normal']
   * @param {AbortSignal} [opts.signal]
   * @param {number} [opts.timeoutMs] Run-time limit, measured from when the job starts running.
   * @param {(value: *) => void} [opts.onProgress]
   * @returns {Promise<*>} Never throws synchronously; always settles exactly once.
   */
  submit(type, payload, opts = {}) {
    return new Promise((resolve, reject) => {
      let priority;
      try {
        if (typeof type !== 'string' || type === '') throw new TypeError('job type must be a non-empty string');
        priority = normalisePriority(opts.priority);
      } catch (err) {
        reject(err);
        return;
      }
      if (this._closed) {
        this._counters.rejected++;
        reject(new PoolTerminatedError('Worker pool is terminated'));
        return;
      }
      if (this._circuitOpen) {
        this._counters.rejected++;
        reject(this._circuitError());
        return;
      }
      const signal = opts.signal;
      if (signal && signal.aborted) {
        this._counters.cancelled++;
        reject(new JobAbortedError(signal.reason));
        return;
      }

      const job = {
        id: this._nextId++,
        type,
        payload,
        transfer: Array.isArray(opts.transfer) ? opts.transfer : [],
        priority,
        timeoutMs: opts.timeoutMs ?? this._defaultTimeoutMs,
        onProgress: typeof opts.onProgress === 'function' ? opts.onProgress : null,
        signal: signal || null,
        resolve,
        reject,
        state: 'queued',
        slot: null,
        settled: false,
        submittedAt: this._now(),
        startedAt: 0,
        crashRetries: 0,
        timeoutTimer: null,
        abortListener: null,
      };
      this._counters.submitted++;

      if (signal) {
        job.abortListener = () => this._onAbort(job);
        signal.addEventListener('abort', job.abortListener, { once: true });
      }

      this._queues[priority].push(job);
      this._pump();
      if (job.state === 'queued' && this._queuedCount() > this._maxQueue) {
        if (this._overflow === 'reject') {
          this._removeFromQueue(job);
          this._counters.rejected++;
          this._detach(job);
          job.settled = true;
          reject(new QueueFullError(this._maxQueue));
        } else {
          this._removeFromQueue(job);
          job.state = 'blocked';
          this._blocked.push(job);
        }
      }
    });
  }

  /** Resolves once no job is queued, blocked or running. */
  drain() {
    if (this._isIdle()) return Promise.resolve();
    return new Promise((resolve) => this._drainWaiters.push(resolve));
  }

  /**
   * Shut the pool down. Idempotent; calling it again with `graceful: false`
   * upgrades a pending graceful shutdown to an immediate one.
   * @param {{ graceful?: boolean }} [opts]
   * @returns {Promise<void>}
   */
  terminate({ graceful = false } = {}) {
    if (!this._terminatePromise) {
      this._closed = true;
      this._terminatePromise = new Promise((resolve) => {
        this._terminateState = { resolve, graceful };
      });
      if (graceful) this._checkIdle();
      else this._finishImmediate();
    } else if (!graceful && this._terminateState && this._terminateState.graceful) {
      this._terminateState.graceful = false;
      this._finishImmediate();
    }
    return this._terminatePromise;
  }

  /** Re-close an open circuit breaker so new work is accepted again. */
  resetCircuit() {
    this._circuitOpen = false;
    this._crashTimes = [];
    this._lastCrash = null;
  }

  /** Snapshot of pool counters. Times are in clock milliseconds. */
  stats() {
    let busy = 0;
    for (const slot of this._slots) if (slot.state !== 'idle') busy++;
    return {
      workers: this._slots.size,
      busy,
      queued: this._queuedCount(),
      blocked: this._blocked.length,
      running: this._running,
      submitted: this._counters.submitted,
      completed: this._counters.completed,
      failed: this._counters.failed,
      cancelled: this._counters.cancelled,
      rejected: this._counters.rejected,
      crashed: this._counters.crashed,
      retried: this._counters.retried,
      hardKills: this._counters.hardKills,
      spawned: this._counters.spawned,
      avgWaitMs: this._waitCount ? this._waitTotal / this._waitCount : 0,
      avgRunMs: this._runCount ? this._runTotal / this._runCount : 0,
      circuitOpen: this._circuitOpen,
      terminated: this._closed,
    };
  }

  // ---- internals ---------------------------------------------------------

  _queuedCount() {
    return this._queues[0].length + this._queues[1].length + this._queues[2].length;
  }

  _isIdle() {
    return this._queuedCount() === 0 && this._blocked.length === 0 && this._running === 0;
  }

  _removeFromQueue(job) {
    const q = this._queues[job.priority];
    const i = q.indexOf(job);
    if (i >= 0) q.splice(i, 1);
  }

  _nextQueued() {
    for (const q of this._queues) if (q.length > 0) return q.shift();
    return null;
  }

  _circuitError() {
    return new CircuitOpenError(this._crashTimes.length, this._breakerWindow, this._lastCrash || undefined);
  }

  /** Remove timers and abort listener from a job. */
  _detach(job) {
    if (job.timeoutTimer !== null) {
      this._clearTimeout(job.timeoutTimer);
      job.timeoutTimer = null;
    }
    if (job.abortListener && job.signal) {
      job.signal.removeEventListener('abort', job.abortListener);
      job.abortListener = null;
    }
  }

  /**
   * Settle a job exactly once.
   * @param {object} job
   * @param {'ok'|'fail'|'cancel'} kind
   * @param {*} value
   */
  _settle(job, kind, value) {
    if (job.settled) return;
    job.settled = true;
    this._detach(job);
    const wasRunning = job.state === 'running';
    if (wasRunning) {
      this._running--;
      this._runTotal += this._now() - job.startedAt;
      this._runCount++;
    }
    job.state = 'done';
    if (kind === 'ok') this._counters.completed++;
    else if (kind === 'cancel') this._counters.cancelled++;
    else this._counters.failed++;
    if (kind === 'ok') job.resolve(value);
    else job.reject(value);
  }

  _onAbort(job) {
    if (job.settled) return;
    const error = new JobAbortedError(job.signal ? job.signal.reason : undefined);
    if (job.state === 'running') {
      this._cancelRunning(job, error, 'cancel');
    } else {
      if (job.state === 'queued') this._removeFromQueue(job);
      else if (job.state === 'blocked') this._blocked.splice(this._blocked.indexOf(job), 1);
      this._settle(job, 'cancel', error);
    }
    this._pump();
    this._checkIdle();
  }

  _onTimeout(job) {
    job.timeoutTimer = null;
    if (job.settled || job.state !== 'running') return;
    this._cancelRunning(job, new JobTimeoutError(job.timeoutMs), 'fail');
    this._pump();
    this._checkIdle();
  }

  /** Reject a running job now, then try a cooperative cancel and fall back to killing the worker. */
  _cancelRunning(job, error, kind) {
    const slot = job.slot;
    this._settle(job, kind, error);
    slot.job = null;
    slot.state = 'cancelling';
    slot.cancelId = job.id;
    try {
      slot.worker.postMessage({ id: job.id, cancel: true });
    } catch (_) {
      // A dead worker will be reported through its own error path.
    }
    if (!(this._cancelGraceMs > 0)) {
      this._hardKill(slot);
    } else if (Number.isFinite(this._cancelGraceMs)) {
      slot.graceTimer = this._setTimeout(() => {
        slot.graceTimer = null;
        if (!slot.dead && slot.state === 'cancelling') {
          this._hardKill(slot);
          this._pump();
          this._checkIdle();
        }
      }, this._cancelGraceMs);
    }
  }

  _hardKill(slot) {
    this._counters.hardKills++;
    this._destroySlot(slot);
  }

  _destroySlot(slot) {
    if (slot.dead) return;
    slot.dead = true;
    if (slot.idleTimer !== null) this._clearTimeout(slot.idleTimer);
    if (slot.graceTimer !== null) this._clearTimeout(slot.graceTimer);
    slot.idleTimer = null;
    slot.graceTimer = null;
    this._slots.delete(slot);
    try {
      slot.worker.terminate();
    } catch (_) {
      // Already gone.
    }
  }

  _spawn() {
    let worker;
    try {
      worker = this._factory();
    } catch (err) {
      this._counters.spawned++;
      this._recordCrash(new WorkerCrashError('spawn failed: ' + (err && err.message), err));
      return null;
    }
    this._counters.spawned++;
    const slot = {
      worker,
      state: 'idle',
      job: null,
      cancelId: 0,
      dead: false,
      idleTimer: null,
      graceTimer: null,
    };
    this._slots.add(slot);
    listen(worker, 'message', (ev) => this._onMessage(slot, ev && ev.data));
    listen(worker, 'error', (ev) =>
      this._onCrash(slot, new WorkerCrashError(describeErrorEvent(ev), ev && ev.error))
    );
    listen(worker, 'messageerror', () =>
      this._onCrash(slot, new WorkerCrashError('messageerror: message could not be deserialised'))
    );
    listen(worker, 'exit', (ev) =>
      this._onCrash(slot, new WorkerCrashError(`worker exited unexpectedly (code ${ev && ev.code})`))
    );
    return slot;
  }

  _idleSlot() {
    for (const slot of this._slots) if (slot.state === 'idle' && !slot.dead) return slot;
    return null;
  }

  /** Move jobs onto workers while capacity and breaker allow. */
  _pump() {
    for (;;) {
      if (this._circuitOpen) return;
      this._admitBlocked();
      if (this._queuedCount() === 0) return;
      let slot = this._idleSlot();
      if (!slot) {
        if (this._slots.size >= this._maxWorkers) return;
        slot = this._spawn();
        if (!slot) {
          if (this._circuitOpen) return;
          // Spawn failed but the breaker is still closed: fail one job rather than spin.
          this._settle(this._nextQueued(), 'fail', this._lastCrash);
          continue;
        }
      }
      const job = this._nextQueued();
      this._dispatch(slot, job);
    }
  }

  _admitBlocked() {
    while (this._blocked.length > 0 && this._queuedCount() < this._maxQueue) {
      const job = this._blocked.shift();
      job.state = 'queued';
      this._queues[job.priority].push(job);
    }
  }

  _dispatch(slot, job) {
    if (slot.idleTimer !== null) {
      this._clearTimeout(slot.idleTimer);
      slot.idleTimer = null;
    }
    job.state = 'running';
    job.slot = slot;
    job.startedAt = this._now();
    this._running++;
    this._waitTotal += job.startedAt - job.submittedAt;
    this._waitCount++;
    slot.job = job;
    slot.state = 'busy';
    if (job.timeoutMs > 0 && Number.isFinite(job.timeoutMs)) {
      job.timeoutTimer = this._setTimeout(() => this._onTimeout(job), job.timeoutMs);
    }
    try {
      slot.worker.postMessage({ id: job.id, type: job.type, payload: job.payload }, job.transfer);
    } catch (err) {
      // Job-level failure (e.g. DataCloneError): the worker itself is fine.
      slot.job = null;
      slot.state = 'idle';
      this._settle(job, 'fail', err);
      this._armIdle(slot);
    }
  }

  _armIdle(slot) {
    if (
      this._idleTimeoutMs > 0 &&
      slot.idleTimer === null &&
      !slot.dead &&
      slot.state === 'idle'
    ) {
      slot.idleTimer = this._setTimeout(() => {
        slot.idleTimer = null;
        if (!slot.dead && slot.state === 'idle' && this._slots.size > this._minWorkers) {
          this._destroySlot(slot);
        }
      }, this._idleTimeoutMs);
    }
  }

  _releaseSlot(slot) {
    slot.job = null;
    slot.state = 'idle';
    this._armIdle(slot);
    this._pump();
    this._checkIdle();
  }

  _onMessage(slot, data) {
    if (slot.dead || data === null || typeof data !== 'object') return;
    const id = data.id;
    if (slot.state === 'cancelling') {
      if (id === slot.cancelId && (data.cancelled === true || typeof data.ok === 'boolean')) {
        if (slot.graceTimer !== null) {
          this._clearTimeout(slot.graceTimer);
          slot.graceTimer = null;
        }
        this._releaseSlot(slot);
      }
      return;
    }
    const job = slot.job;
    if (!job || job.settled || job.id !== id) return;
    if (data.ok === true) {
      this._settle(job, 'ok', data.result);
      this._releaseSlot(slot);
    } else if (data.ok === false) {
      this._settle(job, 'fail', deserializeError(data.error));
      this._releaseSlot(slot);
    } else if (data.cancelled === true) {
      this._settle(job, 'cancel', new JobAbortedError('cancelled by worker'));
      this._releaseSlot(slot);
    } else if ('progress' in data) {
      if (job.onProgress) {
        try {
          job.onProgress(data.progress);
        } catch (_) {
          // A throwing progress callback must not break the pool.
        }
      }
    }
  }

  _recordCrash(error) {
    const t = this._now();
    this._counters.crashed++;
    this._lastCrash = error;
    if (this._breakerMax > 0) {
      this._crashTimes.push(t);
      const cutoff = t - this._breakerWindow;
      this._crashTimes = this._crashTimes.filter((c) => c > cutoff);
    }
  }

  _onCrash(slot, error) {
    if (slot.dead) return;
    const job = slot.job;
    this._destroySlot(slot);
    this._recordCrash(error);
    if (job && !job.settled) {
      job.slot = null;
      if (job.crashRetries < this._retries && job.transfer.length === 0 && !this._closed) {
        // Requeue at the front of its level; the failed attempt's timers are dropped.
        this._running--;
        job.state = 'queued';
        if (job.timeoutTimer !== null) {
          this._clearTimeout(job.timeoutTimer);
          job.timeoutTimer = null;
        }
        job.crashRetries++;
        this._counters.retried++;
        this._queues[job.priority].unshift(job);
      } else {
        this._settle(job, 'fail', error);
      }
    }
    this._checkBreaker();
    this._pump();
    this._checkIdle();
  }

  _checkBreaker() {
    if (this._circuitOpen || this._breakerMax <= 0) return;
    if (this._crashTimes.length >= this._breakerMax) {
      this._circuitOpen = true;
      const pending = [...this._queues[0], ...this._queues[1], ...this._queues[2], ...this._blocked];
      this._queues = [[], [], []];
      this._blocked = [];
      for (const job of pending) this._settle(job, 'fail', this._circuitError());
    }
  }

  _checkIdle() {
    if (!this._isIdle()) return;
    const waiters = this._drainWaiters;
    this._drainWaiters = [];
    for (const resolve of waiters) resolve();
    const st = this._terminateState;
    if (st && st.graceful) this._finishImmediate();
  }

  _finishImmediate() {
    const st = this._terminateState;
    const pending = [
      ...this._queues[0],
      ...this._queues[1],
      ...this._queues[2],
      ...this._blocked,
    ];
    this._queues = [[], [], []];
    this._blocked = [];
    const running = [];
    for (const slot of this._slots) if (slot.job) running.push(slot.job);
    for (const job of pending) this._settle(job, 'cancel', new PoolTerminatedError());
    for (const job of running) this._settle(job, 'cancel', new PoolTerminatedError());
    for (const slot of [...this._slots]) this._destroySlot(slot);
    const waiters = this._drainWaiters;
    this._drainWaiters = [];
    for (const resolve of waiters) resolve();
    if (st) st.resolve();
  }
}

/** Result wrapper carrying a transfer list; see `withTransfer`. */
export class TransferableResult {
  constructor(result, transfer) {
    this.result = result;
    this.transfer = transfer;
  }
}

/**
 * Return a handler result together with buffers to transfer back to the pool.
 * @param {*} result
 * @param {Transferable[]} transfer
 */
export function withTransfer(result, transfer) {
  return new TransferableResult(result, transfer);
}

function unwrapResult(value) {
  if (value instanceof TransferableResult) return { result: value.result, transfer: value.transfer };
  if (
    value !== null &&
    typeof value === 'object' &&
    Object.getPrototypeOf(value) === Object.prototype &&
    Array.isArray(value.transfer) &&
    Object.keys(value).length === 2 &&
    'result' in value
  ) {
    return { result: value.result, transfer: value.transfer };
  }
  return { result: value, transfer: [] };
}

/**
 * Worker-side half of the protocol. Call once inside a worker script.
 *
 * Handlers receive `(payload, ctx)` where `ctx` is
 * `{ id, type, signal, progress(value), transfer(...buffers) }`:
 *  - `ctx.signal` aborts when the pool sends a cancel request (cooperative cancel);
 *  - `ctx.progress(value)` posts a progress message;
 *  - `ctx.transfer(...)` adds buffers to the result's transfer list; alternatively
 *    return `withTransfer(result, list)` or a plain `{ result, transfer }`.
 * A cancelled job is acknowledged with `{ id, cancelled: true }` once its handler
 * settles; its result is discarded.
 *
 * @param {Record<string, (payload: *, ctx: object) => *>} handlers
 * @param {{ scope?: { addEventListener: Function, postMessage: Function } }} [options]
 * @returns {() => void} Function that detaches the message listener.
 */
export function serveWorkerPool(handlers, options = {}) {
  const scope = options.scope || globalThis;
  const active = new Map();

  function post(message, transfer) {
    try {
      scope.postMessage(message, transfer && transfer.length ? transfer : []);
    } catch (err) {
      // e.g. DataCloneError on the result: report it as a job error instead.
      try {
        scope.postMessage({ id: message.id, ok: false, error: serializeError(err) }, []);
      } catch (_) {
        // Nothing more can be done.
      }
    }
  }

  async function run(id, type, payload) {
    const controller = new AbortController();
    const extra = [];
    active.set(id, controller);
    const ctx = {
      id,
      type,
      signal: controller.signal,
      progress(value) {
        if (!controller.signal.aborted) post({ id, progress: value });
      },
      transfer(...buffers) {
        extra.push(...buffers.flat());
      },
    };
    let outcome;
    try {
      const handler = Object.prototype.hasOwnProperty.call(handlers, type) ? handlers[type] : null;
      if (typeof handler !== 'function') throw new TypeError(`Unknown job type: ${type}`);
      const value = unwrapResult(await handler(payload, ctx));
      outcome = { ok: true, result: value.result, transfer: [...extra, ...value.transfer] };
    } catch (err) {
      outcome = { ok: false, error: serializeError(err) };
    }
    active.delete(id);
    if (controller.signal.aborted) {
      post({ id, cancelled: true });
    } else if (outcome.ok) {
      post({ id, ok: true, result: outcome.result }, outcome.transfer);
    } else {
      post({ id, ok: false, error: outcome.error });
    }
  }

  function onMessage(ev) {
    const data = ev && ev.data;
    if (data === null || typeof data !== 'object' || data.id === undefined) return;
    if (data.cancel === true) {
      const controller = active.get(data.id);
      if (controller) controller.abort();
      else post({ id: data.id, cancelled: true });
      return;
    }
    if (typeof data.type !== 'string' || active.has(data.id)) return;
    run(data.id, data.type, data.payload);
  }

  scope.addEventListener('message', onMessage);
  return () => {
    if (typeof scope.removeEventListener === 'function') scope.removeEventListener('message', onMessage);
  };
}
