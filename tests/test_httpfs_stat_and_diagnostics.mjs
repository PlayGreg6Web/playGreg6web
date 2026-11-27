import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { CrashReporter, formatGitHubIssueMarkdown } from '../web/telemetry/crash_report.js';

const require = createRequire(import.meta.url);
const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const ioWorkerMod = require('../web/workers/io_worker.js');
const loaderMod = require('../web/loader.js');

const {
  HttpfsOp,
  HttpfsStatus,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxHost,
  HttpfsMailboxClient,
  HttpfsMailboxPoolHost,
  HttpfsMailboxPoolClient,
} = mailboxMod;

const { IoWorkerController, HTTPFS_OP_WRITE } = ioWorkerMod;
const {
  createHostImports,
  sanitizeIoWorkerStats,
  readRuntimeDiagnostics,
  handleIoWorkerMessage,
  createPatchedPthreadEnv,
} = loaderMod;

test('HttpfsOp.Stat is opcode 5 and does not collide with HTTPFS_OP_WRITE (4)', () => {
  assert.equal(HTTPFS_OP_WRITE, 4);
  assert.equal(HttpfsOp.Stat, 5);
  assert.equal(HttpfsOp.STAT, 5);
  assert.notEqual(HttpfsOp.Stat, HTTPFS_OP_WRITE);
});

test('HttpfsMailboxHost answers Stat from manifest in 1 roundtrip with 0 fetches', async () => {
  const sab = createHttpfsMailboxBuffer();
  let fetchCalls = 0;
  const host = new HttpfsMailboxHost(sab, {
    manifest: {
      files: [
        { path: 'game/common.rpf', size: 1234567890 },
        { path: 'update/x64/dlc.rpf', size: 4294967297 }, // > 4 GiB
        { path: 'empty.bin', size: 0 },
      ],
    },
    fetchImpl: async () => {
      fetchCalls += 1;
      throw new Error('fetch should not be called for manifest-backed Stat');
    },
  });
  const client = new HttpfsMailboxClient(sab);
  host.start();

  try {
    assert.equal(await client.statAsync('game/common.rpf'), 1234567890);
    assert.equal(await client.statAsync('update/x64/dlc.rpf'), 4294967297);
    assert.equal(await client.statAsync('empty.bin'), 0);
    assert.equal(await client.statAsync('does/not/exist.rpf'), -2);
    assert.equal(await client.statAsync('../escape.rpf'), -22);
    assert.equal(await client.statAsync(''), -22);
    assert.equal(fetchCalls, 0);
  } finally {
    await host.stop();
  }
});

test('HttpfsMailboxPoolHost + HttpfsMailboxPoolClient support concurrent statAsync', async () => {
  const sabs = createHttpfsMailboxPool(4);
  const host = new HttpfsMailboxPoolHost(sabs, {
    manifest: {
      files: [
        { path: 'a.rpf', size: 1024 },
        { path: 'b.rpf', size: 2048 },
        { path: 'c.rpf', size: 4096 },
      ],
    },
  });
  const poolClient = new HttpfsMailboxPoolClient(sabs);
  host.start();

  try {
    const results = await Promise.all([
      poolClient.statAsync('a.rpf'),
      poolClient.statAsync('b.rpf'),
      poolClient.statAsync('c.rpf'),
      poolClient.statAsync('missing.rpf'),
    ]);
    assert.deepEqual(results, [1024, 2048, 4096, -2]);
  } finally {
    await host.stop();
  }
});

test('IoWorkerController serves HttpfsOp.Stat from manifest/caches with 0 network fetches and exposes mailboxAttached', async () => {
  const fetchCalls = [];
  const controller = new IoWorkerController({
    dataBase: '/data/',
    gameBase: '/game/',
    manifest: {
      files: [
        { path: 'common.rpf', size: 5368709120 }, // 5 GiB
        { path: 'x64/audio/sfx.awc', size: 987654 },
      ],
    },
    fetchImpl: async (url, init = {}) => {
      fetchCalls.push({ url, method: init.method || 'GET', range: init.headers?.Range || null });
      if (url === '/game/unlisted.rpf') {
        if ((init.method || 'GET') === 'HEAD') {
          return {
            ok: true,
            status: 200,
            headers: new Map([['content-length', '77777']]),
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          };
        }
        return {
          ok: true,
          status: 206,
          headers: new Map([['content-range', 'bytes 0-0/77777']]),
          async arrayBuffer() {
            return new Uint8Array([0x52]).buffer;
          },
        };
      }
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    },
  });

  assert.equal(controller.getStats().mailboxAttached, false);

  const sabs = createHttpfsMailboxPool(2);
  await controller.attachMailbox(sabs);
  const client = new HttpfsMailboxPoolClient(sabs);

  try {
    const statsAfterAttach = controller.getStats();
    assert.equal(statsAfterAttach.mailboxAttached, true);
    assert.equal(statsAfterAttach.mailboxLanes, 2);

    // Manifest hits (including Xbox G:\ prefix, game/ prefix, and case-insensitive match): 0 fetches.
    assert.equal(await client.statAsync('game/common.rpf'), 5368709120);
    assert.equal(await client.statAsync('G:\\Common.RPF'), 5368709120);
    assert.equal(await client.statAsync('x64/audio/sfx.awc'), 987654);
    // Missing file on /data/ mount proven absent by manifest: instant ENOENT, 0 fetches.
    assert.equal(await client.statAsync('x64/audio/missing.awc'), -2);
    assert.equal(fetchCalls.length, 0);

    // Unindexed /game/ file falls back to 1 network probe (not binary search) and caches metadata.
    assert.equal(await client.statAsync('game/unlisted.rpf'), 77777);
    const fetchesAfterFirst = fetchCalls.length;
    assert.ok(fetchesAfterFirst >= 1 && fetchesAfterFirst <= 2);

    // Second stat on unlisted.rpf hits cached metadata with 0 additional fetches.
    assert.equal(await client.statAsync('game/unlisted.rpf'), 77777);
    assert.equal(fetchCalls.length, fetchesAfterFirst);

    const statsAfterStat = controller.getStats();
    assert.equal(statsAfterStat.statHits, 5);
    assert.equal(statsAfterStat.statMisses, 1);
  } finally {
    await controller.detachMailbox();
    assert.equal(controller.getStats().mailboxAttached, false);
  }
});

test('loader createHostImports exposes host_httpfs_stat and pthread env patching wires it', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1 });
  const bytes = new TextEncoder().encode('game/common.rpf');
  new Uint8Array(memory.buffer).set(bytes, 128);

  const statCalls = [];
  const hostImports = createHostImports({
    memory,
    httpfsClient: {
      open: () => 1,
      read: () => 0,
      close: () => true,
      stat: (p) => {
        statCalls.push(p);
        return p === 'game/common.rpf' ? 9876543210 : -2;
      },
    },
  });

  assert.equal(typeof hostImports.env.host_httpfs_stat, 'function');
  assert.equal(hostImports.env.host_httpfs_stat(128n, BigInt(bytes.byteLength)), 9876543210n);
  assert.deepEqual(statCalls, ['game/common.rpf']);

  // Patched pthread env replaces stub host_httpfs_stat with the real host import.
  const stubStat = () => -1n;
  stubStat.stub = true;
  const patched = createPatchedPthreadEnv(
    { host_httpfs_stat: stubStat, keepMe: 123 },
    hostImports.env,
    new Set(['host_httpfs_stat']),
  );
  assert.equal(patched.host_httpfs_stat, hostImports.env.host_httpfs_stat);
  assert.equal(patched.keepMe, 123);
});

test('loader handleIoWorkerMessage surfaces mailboxAttached and ioWorkerStats instead of dropping them', () => {
  const forwarded = [];
  const progress = [];
  const readyCalls = [];
  let statsRequested = 0;
  const state = { mailboxAttached: false, mailboxLanes: 0, lastStats: null };
  const hooks = {
    state,
    onWorkerReady: (name, summary) => readyCalls.push({ name, summary }),
    postToMain: (msg) => forwarded.push(msg),
    postProgress: (msg) => progress.push(msg),
    requestStats: () => {
      statsRequested += 1;
    },
  };

  assert.equal(handleIoWorkerMessage({ ready: true, summary: '12 files indexed' }, hooks), true);
  assert.deepEqual(readyCalls, [{ name: 'io_worker', summary: '12 files indexed' }]);

  assert.equal(handleIoWorkerMessage({ mailboxAttached: true, mailboxLanes: 8 }, hooks), true);
  assert.equal(state.mailboxAttached, true);
  assert.equal(state.mailboxLanes, 8);
  assert.equal(statsRequested, 1);
  assert.equal(forwarded.length, 1);
  assert.equal(forwarded[0].type, 'io-stats');
  assert.equal(forwarded[0].mailboxAttached, true);
  assert.equal(forwarded[0].mailboxLanes, 8);
  assert.equal(forwarded[0].ioWorkerStats.mailboxAttached, true);
  assert.equal(forwarded[0].ioWorkerStats.mailboxLanes, 8);

  assert.equal(
    handleIoWorkerMessage(
      {
        stats: {
          mailboxAttached: true,
          mailboxLanes: 8,
          readsCompleted: 420,
          bytesTransferred: 10485760,
          errorCount: 2,
          cacheHits: 380,
          cacheMisses: 40,
          openHandles: 6,
          statHits: 55,
          statMisses: 1,
          mailbox: { activeLanes: 3, peakActiveLanes: 8, totalRequests: 500, syncFastPathHits: 350 },
        },
      },
      hooks,
    ),
    true,
  );
  assert.equal(forwarded.length, 2);
  assert.equal(forwarded[1].ioWorkerStats.readsCompleted, 420);
  assert.equal(forwarded[1].ioWorkerStats.bytesTransferred, 10485760);
  assert.equal(forwarded[1].ioWorkerStats.inFlightRequests, 3);
  assert.equal(forwarded[1].ioWorkerStats.errorCount, 2);
  assert.equal(forwarded[1].ioWorkerStats.cacheHits, 380);
  assert.equal(forwarded[1].ioWorkerStats.statHits, 55);

  assert.equal(handleIoWorkerMessage({ error: 'range timeout' }, hooks), true);
  assert.deepEqual(progress, [{ error: 'io_worker: range timeout' }]);
});

test('loader readRuntimeDiagnostics decodes playgta6_runtime_diagnostics_json from Wasm memory', () => {
  const payload = JSON.stringify({
    dstorage: {
      requests_completed: 120,
      requests_failed: 0,
      bytes_delivered: 6553600,
      debug_state: 'worker idle after 14 batches',
    },
    blocked_waits: [],
    streaming_drain: { available: true, owner: '0x14764b240', all_sum: 3, all: [1, 2, 0, 0, 0] },
  });
  const encoded = new TextEncoder().encode(payload + '\0');
  const heap = new Uint8Array(4096);
  heap.set(encoded, 256);

  const mockModule = {
    HEAPU8: heap,
    _playgta6_runtime_diagnostics_json: () => 256n,
  };
  const diag = readRuntimeDiagnostics(mockModule);
  assert.equal(diag.dstorage.requests_completed, 120);
  assert.equal(diag.dstorage.bytes_delivered, 6553600);
  assert.equal(diag.streaming_drain.available, true);
  assert.equal(diag.streaming_drain.all_sum, 3);
  assert.deepEqual(diag.blocked_waits, []);
});

test('CrashReporter exports sanitized ioWorkerStats and runtimeDiagnostics in JSON and Markdown', () => {
  const reporter = new CrashReporter({ now: () => 1700000000000 });
  const storedIo = reporter.setIoWorkerStats({
    mailboxAttached: true,
    mailboxLanes: 8,
    readsCompleted: 512,
    bytesTransferred: 33554432,
    inFlightRequests: 2,
    errorCount: 1,
    cacheHits: 490,
    cacheMisses: 22,
    statHits: 80,
    statMisses: 0,
    secretPath: 'C:\\Users\\alice\\game_scarlett_master.exe',
  });
  assert.equal(storedIo.mailboxAttached, true);
  assert.equal(storedIo.readsCompleted, 512);
  assert.equal('secretPath' in storedIo, false);

  const storedRt = reporter.setRuntimeDiagnostics({
    dstorage: {
      requests_completed: 77,
      requests_failed: 0,
      bytes_delivered: 4194304,
      debug_state: 'worker idle after 9 batches',
    },
    blocked_waits: [{ wait_ms: 2500, kind: 'event' }],
    streaming_drain: { available: true, all_sum: 2, pending_categories: [{ index: 4, counters: [2, 0, 0, 0, 0] }] },
  });
  assert.equal(storedRt.dstorage.requests_completed, 77);

  const report = reporter.buildReport();
  assert.equal(report.bootDiagnostics.ioWorkerStats.mailboxAttached, true);
  assert.equal(report.bootDiagnostics.ioWorkerStats.readsCompleted, 512);
  assert.equal(report.bootDiagnostics.ioWorkerStats.bytesTransferred, 33554432);
  assert.equal(report.bootDiagnostics.ioWorkerStats.inFlightRequests, 2);
  assert.equal(report.bootDiagnostics.ioWorkerStats.errorCount, 1);
  assert.equal(report.bootDiagnostics.ioWorkerStats.cacheHits, 490);
  assert.equal(report.bootDiagnostics.runtimeDiagnostics.dstorage.requests_completed, 77);
  assert.equal(report.bootDiagnostics.runtimeDiagnostics.streaming_drain.all_sum, 2);

  const md = formatGitHubIssueMarkdown(report);
  assert.ok(md.includes('**IO / HTTPFS:**'));
  assert.ok(md.includes('mailboxAttached=yes'));
  assert.ok(md.includes('reads=512'));
  assert.ok(md.includes('cacheHits=490'));
  assert.ok(!md.includes('alice'));
});
