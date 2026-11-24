import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const rpfMod = require('../web/workers/rpf9_vfs.js');

const {
  IoWorkerController,
  createIoWorkerHandler,
  resolveHttpfsRoute,
  IO_PRIORITY,
} = ioWorkerMod;

const {
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  HttpfsMailboxClient,
} = mailboxMod;

const {
  RPF9_MAGIC_LE,
  RPF_ENCRYPTION_OPEN,
  buildSyntheticRpf9Buffer,
} = rpfMod;

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder('utf-8');

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 53) ^ ((i >> 6) * 29) ^ seed) & 0xff;
  }
  return out;
}

/**
 * Multi-base mock Range fetch keyed by exact URL path (e.g. `/boot/program-manifest.json`,
 * `/game/appdata0.rpf`, `/data/data-manifest.json`), with optional per-URL status overrides.
 */
function createMultiBaseMockFetch(urlBytesMap, options = {}) {
  const requests = [];
  const statusOverrides = options.statusOverrides || new Map();

  async function fetchImpl(url, init = {}) {
    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    requests.push({ url, rangeHeader });

    const overrideStatus = statusOverrides.get(url);
    if (overrideStatus) {
      return {
        ok: false,
        status: overrideStatus,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error(`HTTP ${overrideStatus} for ${url}`);
        },
      };
    }

    const data = urlBytesMap.get(url);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error(`HTTP 404 for ${url}`);
        },
      };
    }

    if (!rangeHeader) {
      return {
        ok: true,
        status: 200,
        headers: new Map([['Content-Length', String(data.byteLength)]]),
        async arrayBuffer() {
          return data.slice().buffer;
        },
        async json() {
          return JSON.parse(utf8Decoder.decode(data));
        },
      };
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    assert.ok(match, `Unexpected Range header: ${rangeHeader}`);
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);
    if (reqStart >= data.byteLength) {
      return {
        ok: false,
        status: 416,
        headers: new Map([['Content-Range', `bytes */${data.byteLength}`]]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.slice(reqStart, actualEnd + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return { fetchImpl, requests };
}

async function rawMailboxOpen(sab, relPath) {
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, 40);
  const payloadView = new Uint8Array(sab, 40, sab.byteLength - 40);

  const encoded = utf8Encoder.encode(relPath);
  headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Open, true);
  headerView.setInt32(HttpfsOffsets.HANDLE, -1, true);
  headerView.setBigUint64(HttpfsOffsets.OFFSET, 0n, true);
  headerView.setBigUint64(HttpfsOffsets.LENGTH, BigInt(encoded.byteLength), true);
  headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
  payloadView.set(encoded, 0);

  Atomics.store(statusView, 0, HttpfsStatus.RequestReady);
  Atomics.notify(statusView, 0, 1);

  let cur = Atomics.load(statusView, 0);
  while (cur === HttpfsStatus.RequestReady) {
    const waitObj = Atomics.waitAsync(statusView, 0, HttpfsStatus.RequestReady, 5000);
    const res = waitObj.async ? await waitObj.value : waitObj.value;
    cur = Atomics.load(statusView, 0);
    if (res === 'timed-out' && cur === HttpfsStatus.RequestReady) {
      throw new Error(`Mailbox hung during OPEN(${relPath})`);
    }
  }

  const result = headerView.getBigInt64(HttpfsOffsets.RESULT, true);
  Atomics.store(statusView, 0, HttpfsStatus.Idle);
  Atomics.notify(statusView, 0, 1);

  return { status: cur, result };
}

async function startMultiBaseHttpServer(routesByPath) {
  const requests = [];

  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');
    const pathname = reqUrl.pathname;
    const rangeHeader = req.headers.range || null;
    requests.push({ method: req.method, pathname, rangeHeader });

    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Accept-Ranges', 'bytes');

    const data = routesByPath.get(pathname);
    if (!data) {
      res.writeHead(404);
      res.end();
      return;
    }

    if (!rangeHeader) {
      res.writeHead(200, { 'Content-Length': String(data.byteLength) });
      res.end(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
      return;
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    if (!match) {
      res.writeHead(400);
      res.end();
      return;
    }

    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);
    if (reqStart >= data.byteLength) {
      res.writeHead(416, { 'Content-Range': `bytes */${data.byteLength}` });
      res.end();
      return;
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.subarray(reqStart, actualEnd + 1);
    res.writeHead(206, {
      'Content-Range': `bytes ${reqStart}-${actualEnd}/${data.byteLength}`,
      'Content-Length': String(slice.byteLength),
    });
    res.end(Buffer.from(slice.buffer, slice.byteOffset, slice.byteLength));
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });

  const addr = server.address();
  const origin = `http://127.0.0.1:${addr.port}`;
  return {
    origin,
    requests,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

// ---------------------------------------------------------------------------
// 1. resolveHttpfsRoute unit tests: Xbox/Win32 normalization & security
// ---------------------------------------------------------------------------

test('1. resolveHttpfsRoute strips Xbox/Win32 drive prefixes (G:\\, T:\\, D:\\, C:\\, \\\\?\\G:\\, \\\\?\\D:\\) and normalizes backslashes', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  assert.deepEqual(resolveHttpfsRoute('G:\\appdata0.rpf', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'appdata0.rpf',
    url: '/game/appdata0.rpf',
  });

  assert.deepEqual(resolveHttpfsRoute('T:\\x64\\common.rpf', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'x64/common.rpf',
    url: '/game/x64/common.rpf',
  });

  assert.deepEqual(resolveHttpfsRoute('D:/assets/splash.png', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'assets/splash.png',
    url: '/game/assets/splash.png',
  });

  assert.deepEqual(resolveHttpfsRoute('C:\\Game\\MicrosoftGame.Config', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'Game/MicrosoftGame.Config',
    url: '/game/Game/MicrosoftGame.Config',
  });

  assert.deepEqual(resolveHttpfsRoute('\\\\?\\G:\\audio\\sfx.ttpf', bases), {
    ok: true,
    isRpf: false,
    mount: 'game',
    base: '/game/',
    relPath: 'audio/sfx.ttpf',
    url: '/game/audio/sfx.ttpf',
  });

  assert.deepEqual(resolveHttpfsRoute('\\\\?\\D:\\boot\\main\\image\\guest.pgim', bases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: '/boot/',
    relPath: 'main/image/guest.pgim',
    url: '/boot/main/image/guest.pgim',
  });

  assert.deepEqual(resolveHttpfsRoute('g:\\sub\\settings.xml', bases), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'sub/settings.xml',
    url: '/data/sub/settings.xml',
  });
});

test('2. resolveHttpfsRoute strips leading /, ./, and redundant separators while preserving filenames containing double dots', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  assert.equal(resolveHttpfsRoute('./boot/program-manifest.json', bases).url, '/boot/program-manifest.json');
  assert.equal(resolveHttpfsRoute('///./boot//main/./image/guest.pgim', bases).url, '/boot/main/image/guest.pgim');
  assert.equal(resolveHttpfsRoute('.\\x64\\\\core.dll', bases).url, '/game/x64/core.dll');

  const doubleDotInName = resolveHttpfsRoute('x64/level..01.rpf', bases);
  assert.equal(doubleDotInName.ok, true);
  assert.equal(doubleDotInName.relPath, 'x64/level..01.rpf');
  assert.equal(doubleDotInName.url, '/game/x64/level..01.rpf');
});

test('3. resolveHttpfsRoute strictly rejects .. path traversal across POSIX, Windows, Xbox drive, and percent-encoded inputs', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };
  const badPaths = [
    '..',
    '../secret.rpf',
    '../../etc/passwd',
    'boot/../../etc/passwd',
    'boot/main/../../../secret.pgim',
    'G:\\..\\Windows\\System32\\kernel32.dll',
    'G:..\\secret.bin',
    '\\\\?\\D:\\sub\\..\\..\\escape.rpf',
    'T:\\common\\..\\secret.rpf',
    './../escape.json',
    'sub/..',
    'boot/%2e%2e/secret.json',
    'game/%2E%2E/secret.rpf',
  ];

  for (const p of badPaths) {
    assert.deepEqual(
      resolveHttpfsRoute(p, bases),
      { ok: false, error: 'path traversal blocked' },
      `Expected traversal rejection for ${JSON.stringify(p)}`,
    );
  }
});

test('4. resolveHttpfsRoute strictly rejects NUL bytes (\\0) in regular, Xbox, and rpf:// paths', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };
  const nulPaths = [
    'boot/program-manifest.json\0',
    'G:\\appdata0.rpf\0.txt',
    '\\0\0',
    'data-manifest.json\0',
    'rpf://appdata0.rpf/config/settings.xml\0',
    'boot/%00/manifest.json',
  ];

  for (const p of nulPaths) {
    assert.deepEqual(
      resolveHttpfsRoute(p, bases),
      { ok: false, error: 'path traversal blocked' },
      `Expected NUL rejection for ${JSON.stringify(p)}`,
    );
  }
});

test('5. resolveHttpfsRoute rejects empty, non-string, and drive-only / slash-only paths', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  for (const bad of ['', null, undefined, 42, '/', '///', './', '.', 'G:', 'G:\\', '\\\\?\\G:\\']) {
    const res = resolveHttpfsRoute(bad, bases);
    assert.equal(res.ok, false, `Expected failure for ${JSON.stringify(bad)}`);
    assert.equal(typeof res.error, 'string');
  }
});

// ---------------------------------------------------------------------------
// 2. Multi-base routing rules (/boot/, /game/, /data/, rpf://)
// ---------------------------------------------------------------------------

test('6. resolveHttpfsRoute routes boot/..., program-manifest.json, and *.pgim against bootBase (relative & absolute)', () => {
  const relBases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };
  const absBases = {
    dataBase: 'http://127.0.0.1:8080/data',
    bootBase: 'http://127.0.0.1:8080/boot',
    gameBase: 'http://127.0.0.1:8080/game',
  };

  assert.deepEqual(resolveHttpfsRoute('boot/program-manifest.json', relBases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: '/boot/',
    relPath: 'program-manifest.json',
    url: '/boot/program-manifest.json',
  });

  assert.deepEqual(resolveHttpfsRoute('/boot/main/image/guest.pgim', relBases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: '/boot/',
    relPath: 'main/image/guest.pgim',
    url: '/boot/main/image/guest.pgim',
  });

  assert.deepEqual(resolveHttpfsRoute('program-manifest.json', absBases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: 'http://127.0.0.1:8080/boot/',
    relPath: 'program-manifest.json',
    url: 'http://127.0.0.1:8080/boot/program-manifest.json',
  });

  assert.deepEqual(resolveHttpfsRoute('main/image/guest.pgim', absBases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: 'http://127.0.0.1:8080/boot/',
    relPath: 'main/image/guest.pgim',
    url: 'http://127.0.0.1:8080/boot/main/image/guest.pgim',
  });

  assert.deepEqual(resolveHttpfsRoute('boot/chunk_01.wasm', absBases), {
    ok: true,
    isRpf: false,
    mount: 'boot',
    base: 'http://127.0.0.1:8080/boot/',
    relPath: 'chunk_01.wasm',
    url: 'http://127.0.0.1:8080/boot/chunk_01.wasm',
  });
});

test('7. resolveHttpfsRoute routes all game package file types and explicit game/... paths against gameBase', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };
  const cases = [
    ['appdata0.rpf', 'appdata0.rpf'],
    ['x64/audio/dialogue.ttpf', 'x64/audio/dialogue.ttpf'],
    ['xgameruntime.dll', 'xgameruntime.dll'],
    ['Assets/SplashScreen.PNG', 'Assets/SplashScreen.PNG'],
    ['MicrosoftGame.Config', 'MicrosoftGame.Config'],
    ['sub/microsoftgame.config', 'sub/microsoftgame.config'],
    ['package.xvd', 'package.xvd'],
    ['title.xvc', 'title.xvc'],
    ['shaders/cache.bin', 'shaders/cache.bin'],
    ['resources.pri', 'resources.pri'],
    ['/game/appdata0.rpf', 'appdata0.rpf'],
    ['game/custom/file.txt', 'custom/file.txt'],
  ];

  for (const [input, expectedRel] of cases) {
    const res = resolveHttpfsRoute(input, bases);
    assert.deepEqual(
      res,
      {
        ok: true,
        isRpf: false,
        mount: 'game',
        base: '/game/',
        relPath: expectedRel,
        url: `/game/${expectedRel}`,
      },
      `Unexpected route for ${input}`,
    );
  }
});

test('8. resolveHttpfsRoute falls back cleanly to dataBase when bootBase or gameBase is absent', () => {
  const dataOnly = { dataBase: '/data/' };

  assert.deepEqual(resolveHttpfsRoute('boot/program-manifest.json', dataOnly), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'boot/program-manifest.json',
    url: '/data/boot/program-manifest.json',
  });

  assert.deepEqual(resolveHttpfsRoute('main/image/guest.pgim', dataOnly), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'main/image/guest.pgim',
    url: '/data/main/image/guest.pgim',
  });

  assert.deepEqual(resolveHttpfsRoute('G:\\appdata0.rpf', dataOnly), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'appdata0.rpf',
    url: '/data/appdata0.rpf',
  });

  assert.deepEqual(resolveHttpfsRoute('MicrosoftGame.Config', { dataBase: '/data/', bootBase: '/boot/' }), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'MicrosoftGame.Config',
    url: '/data/MicrosoftGame.Config',
  });
});

test('9. resolveHttpfsRoute keeps data-manifest.json, bootset.json, and general relative paths on dataBase', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  assert.deepEqual(resolveHttpfsRoute('data-manifest.json', bases), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'data-manifest.json',
    url: '/data/data-manifest.json',
  });

  assert.deepEqual(resolveHttpfsRoute('bootset.json', bases), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'bootset.json',
    url: '/data/bootset.json',
  });

  assert.deepEqual(resolveHttpfsRoute('shaders/fallback.wgsl.pack', bases), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'shaders/fallback.wgsl.pack',
    url: '/data/shaders/fallback.wgsl.pack',
  });

  assert.deepEqual(resolveHttpfsRoute('config/settings.xml', bases), {
    ok: true,
    isRpf: false,
    mount: 'data',
    base: '/data/',
    relPath: 'config/settings.xml',
    url: '/data/config/settings.xml',
  });
});

test('10. resolveHttpfsRoute preserves rpf:// virtual URIs and rejects traversal inside rpf://', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  assert.deepEqual(resolveHttpfsRoute('rpf://appdata0.rpf/config/settings.xml', bases), {
    ok: true,
    isRpf: true,
    mount: 'rpf',
    base: null,
    relPath: 'rpf://appdata0.rpf/config/settings.xml',
    url: 'rpf://appdata0.rpf/config/settings.xml',
  });

  assert.deepEqual(resolveHttpfsRoute('RPF://update/patch.rpf/sub/file.bin', bases), {
    ok: true,
    isRpf: true,
    mount: 'rpf',
    base: null,
    relPath: 'RPF://update/patch.rpf/sub/file.bin',
    url: 'RPF://update/patch.rpf/sub/file.bin',
  });

  assert.deepEqual(resolveHttpfsRoute('rpf://../escape.rpf/sub/file.bin', bases), {
    ok: false,
    error: 'path traversal blocked',
  });
  assert.deepEqual(resolveHttpfsRoute('rpf://appdata0.rpf/sub/../../secret.xml', bases), {
    ok: false,
    error: 'path traversal blocked',
  });
});

test('11. resolveHttpfsRoute prioritizes explicit boot/ and game/ prefixes over implicit extension heuristics', () => {
  const bases = { dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' };

  // `boot/something.bin` explicitly targets /boot/ even though `.bin` is a game extension.
  const bootBin = resolveHttpfsRoute('boot/something.bin', bases);
  assert.equal(bootBin.mount, 'boot');
  assert.equal(bootBin.relPath, 'something.bin');
  assert.equal(bootBin.url, '/boot/something.bin');

  // `game/notes.txt` explicitly targets /game/ even though `.txt` defaults to /data/.
  const gameTxt = resolveHttpfsRoute('game/notes.txt', bases);
  assert.equal(gameTxt.mount, 'game');
  assert.equal(gameTxt.relPath, 'notes.txt');
  assert.equal(gameTxt.url, '/game/notes.txt');
});

// ---------------------------------------------------------------------------
// 3. IoWorkerController & createIoWorkerHandler wiring tests
// ---------------------------------------------------------------------------

test('12. IoWorkerController.init accepts optional bootBase and gameBase and exposes normalized bases in ready payload', async () => {
  const mock = createMultiBaseMockFetch(new Map());
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    opfsDir: null,
  });

  const ready = await controller.init({
    dataBase: '/data',
    bootBase: '/boot',
    gameBase: '/game',
  });

  assert.equal(ready.ready, true);
  assert.equal(ready.dataBase, '/data/');
  assert.equal(ready.bootBase, '/boot/');
  assert.equal(ready.gameBase, '/game/');
  assert.equal(controller.bootBase, '/boot/');
  assert.equal(controller.gameBase, '/game/');
});

test('13. createIoWorkerHandler handles init (bootBase/gameBase), routed reads, and stats for both raw messages and event objects', async () => {
  const manifestBytes = utf8Encoder.encode('{"program":"gta6","modules":["main"]}');
  const pgimBytes = makePatternBytes(96, 33);
  const rpfBytes = makePatternBytes(128, 77);

  const mock = createMultiBaseMockFetch(
    new Map([
      ['/boot/program-manifest.json', manifestBytes],
      ['/boot/main/image/guest.pgim', pgimBytes],
      ['/game/appdata0.rpf', rpfBytes],
    ]),
  );

  const posted = [];
  const handler = createIoWorkerHandler({
    fetchImpl: mock.fetchImpl,
    postMessage: (msg) => posted.push(msg),
    chunkSize: 64,
    maxBytes: 4096,
    opfsDir: null,
  });

  // Event-style init message (like self.onmessage)
  await handler({
    data: {
      init: true,
      dataBase: '/data/',
      bootBase: '/boot/',
      gameBase: '/game/',
    },
  });
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].bootBase, '/boot/');
  assert.equal(posted[0].gameBase, '/game/');

  // Direct message object read for /boot/program-manifest.json
  await handler({
    read: true,
    id: 1,
    path: 'boot/program-manifest.json',
    offset: 0,
    length: manifestBytes.byteLength,
  });
  assert.equal(posted[1].id, 1);
  assert.deepEqual(new Uint8Array(posted[1].bytes), manifestBytes);

  // Event-style read for Xbox path G:\appdata0.rpf routed to /game/appdata0.rpf
  await handler({
    data: {
      read: true,
      id: 2,
      path: 'G:\\appdata0.rpf',
      offset: 16,
      length: 48,
      priority: IO_PRIORITY.HIGH,
    },
  });
  assert.equal(posted[2].id, 2);
  assert.deepEqual(new Uint8Array(posted[2].bytes), rpfBytes.subarray(16, 64));

  // Verify outbound URLs went to /boot/ and /game/
  const rangeUrls = mock.requests.filter((r) => r.rangeHeader).map((r) => r.url);
  assert.ok(rangeUrls.includes('/boot/program-manifest.json'));
  assert.ok(rangeUrls.includes('/game/appdata0.rpf'));
});

// ---------------------------------------------------------------------------
// 4. HttpfsMailbox end-to-end OPEN / READ / CLOSE over routed URLs
// ---------------------------------------------------------------------------

test('14. HttpfsMailbox OPEN/READ/CLOSE opens /boot/program-manifest.json and /boot/main/image/guest.pgim even when data-manifest.json exists', async () => {
  const dataManifest = {
    abi_version: 1,
    file_count: 1,
    files: [{ path: 'notes.txt', size: 5 }],
  };
  const programManifest = utf8Encoder.encode('{"version":1,"entry":"main/image/guest.pgim"}');
  const pgimData = makePatternBytes(300, 91);

  const mock = createMultiBaseMockFetch(
    new Map([
      ['/data/data-manifest.json', utf8Encoder.encode(JSON.stringify(dataManifest))],
      ['/data/notes.txt', utf8Encoder.encode('hello')],
      ['/boot/program-manifest.json', programManifest],
      ['/boot/main/image/guest.pgim', pgimData],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 128,
    maxBytes: 8192,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(128);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/' });
    await controller.attachMailbox(sab);

    // 1. Open `boot/program-manifest.json` -> routed to `/boot/program-manifest.json`
    const hManifest = await client.openAsync('boot/program-manifest.json');
    assert.ok(hManifest >= 1, `Expected valid handle, got ${hManifest}`);
    assert.equal(controller.getHandleInfo(hManifest).mount, 'boot');
    assert.equal(controller.getHandleInfo(hManifest).url, '/boot/program-manifest.json');

    const manifestOut = new Uint8Array(128);
    const mLen = await client.readAsync(hManifest, 0n, manifestOut, 128);
    assert.equal(mLen, programManifest.byteLength);
    assert.deepEqual(manifestOut.subarray(0, mLen), programManifest);
    assert.equal(await client.closeAsync(hManifest), true);

    // 2. Open `/boot/main/image/guest.pgim` -> routed to `/boot/main/image/guest.pgim`
    const hPgim = await client.openAsync('/boot/main/image/guest.pgim');
    assert.ok(hPgim >= 1);
    const pgimOut = new Uint8Array(512);
    const pLen = await client.readAsync(hPgim, 0n, pgimOut, 512);
    assert.equal(pLen, pgimData.byteLength);
    assert.deepEqual(pgimOut.subarray(0, pLen), pgimData);
    assert.equal(await client.closeAsync(hPgim), true);

    // 3. Also works without leading `boot/` prefix via implicit `*.pgim` routing
    const hImplicit = await client.openAsync('main/image/guest.pgim');
    assert.ok(hImplicit >= 1);
    const sliceOut = new Uint8Array(40);
    const sLen = await client.readAsync(hImplicit, 100n, sliceOut, 40);
    assert.equal(sLen, 40);
    assert.deepEqual(sliceOut, pgimData.subarray(100, 140));
    assert.equal(await client.closeAsync(hImplicit), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('15. HttpfsMailbox OPEN/READ/CLOSE opens /game/ files via G:\\appdata0.rpf, T:\\common.rpf, \\\\?\\D:\\MicrosoftGame.Config, and /game/appdata0.rpf', async () => {
  const appdata = makePatternBytes(220, 11);
  const common = makePatternBytes(180, 22);
  const gameConfig = utf8Encoder.encode('<Game configVersion="1"><Identity Name="GTA6"/></Game>');

  const mock = createMultiBaseMockFetch(
    new Map([
      ['/game/appdata0.rpf', appdata],
      ['/game/common.rpf', common],
      ['/game/MicrosoftGame.Config', gameConfig],
    ]),
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 64,
    maxBytes: 8192,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(128);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    // G:\appdata0.rpf
    const hG = await client.openAsync('G:\\appdata0.rpf');
    assert.ok(hG >= 1);
    assert.equal(controller.getHandleInfo(hG).url, '/game/appdata0.rpf');
    const outG = new Uint8Array(256);
    const lenG = await client.readAsync(hG, 0n, outG, 256);
    assert.equal(lenG, appdata.byteLength);
    assert.deepEqual(outG.subarray(0, lenG), appdata);
    assert.equal(await client.closeAsync(hG), true);

    // T:\common.rpf
    const hT = await client.openAsync('T:\\common.rpf');
    assert.ok(hT >= 1);
    assert.equal(controller.getHandleInfo(hT).url, '/game/common.rpf');
    const outT = new Uint8Array(256);
    const lenT = await client.readAsync(hT, 32n, outT, 100);
    assert.equal(lenT, 100);
    assert.deepEqual(outT.subarray(0, lenT), common.subarray(32, 132));
    assert.equal(await client.closeAsync(hT), true);

    // \\?\D:\MicrosoftGame.Config
    const hCfg = await client.openAsync('\\\\?\\D:\\MicrosoftGame.Config');
    assert.ok(hCfg >= 1);
    const outCfg = new Uint8Array(256);
    const lenCfg = await client.readAsync(hCfg, 0n, outCfg, 256);
    assert.equal(lenCfg, gameConfig.byteLength);
    assert.deepEqual(outCfg.subarray(0, lenCfg), gameConfig);
    assert.equal(await client.closeAsync(hCfg), true);

    // /game/appdata0.rpf hits cached chunks with zero new HTTP requests
    const reqsBefore = mock.requests.length;
    const hDirect = await client.openAsync('/game/appdata0.rpf');
    assert.ok(hDirect >= 1);
    const outDirect = new Uint8Array(64);
    const lenDirect = await client.readAsync(hDirect, 0n, outDirect, 64);
    assert.equal(lenDirect, 64);
    assert.deepEqual(outDirect, appdata.subarray(0, 64));
    assert.equal(mock.requests.length, reqsBefore);
    assert.equal(await client.closeAsync(hDirect), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('16. HttpfsMailbox OPEN rejects .. traversal and NUL bytes with -22 (EINVAL) and zero outbound fetches', async () => {
  const mock = createMultiBaseMockFetch(new Map());
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 64,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(256);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const reqsAfterInit = mock.requests.length;
    const attackPaths = [
      '../secret.rpf',
      'boot/../../etc/passwd',
      'G:\\..\\Windows\\win.ini',
      '\\\\?\\D:\\x64\\..\\..\\secret.dll',
      'appdata0.rpf\0.txt',
      'boot/%2e%2e/secret.json',
    ];

    for (const attack of attackPaths) {
      const res = await rawMailboxOpen(sab, attack);
      assert.equal(res.status, HttpfsStatus.Error, `Expected error for ${attack}`);
      assert.equal(res.result, -22n, `Expected -22 (EINVAL) for ${attack}, got ${res.result}`);
    }

    assert.equal(
      mock.requests.length,
      reqsAfterInit,
      'Traversal/NUL attacks must never issue any outbound HTTP requests',
    );
  } finally {
    await controller.detachMailbox();
  }
});

test('17. HttpfsMailbox OPEN maps HTTP 404 to -2 (ENOENT) and HTTP 403 to -13 (EACCES) while keeping mailbox healthy', async () => {
  const okBytes = utf8Encoder.encode('{"ok":true}');
  const mock = createMultiBaseMockFetch(
    new Map([['/boot/program-manifest.json', okBytes]]),
    {
      statusOverrides: new Map([['/boot/disallowed.exe', 403]]),
    },
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 64,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(256);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
    await controller.attachMailbox(sab);

    const missingBoot = await rawMailboxOpen(sab, 'boot/missing.pgim');
    assert.equal(missingBoot.status, HttpfsStatus.Error);
    assert.equal(missingBoot.result, -2n);

    const missingGame = await rawMailboxOpen(sab, 'G:\\missing.rpf');
    assert.equal(missingGame.status, HttpfsStatus.Error);
    assert.equal(missingGame.result, -2n);

    const forbiddenBoot = await rawMailboxOpen(sab, 'boot/disallowed.exe');
    assert.equal(forbiddenBoot.status, HttpfsStatus.Error);
    assert.equal(forbiddenBoot.result, -13n);

    // Mailbox remains completely healthy after 404/403 errors
    const h = await client.openAsync('program-manifest.json');
    assert.ok(h >= 1);
    const buf = new Uint8Array(32);
    const n = await client.readAsync(h, 0n, buf, 32);
    assert.equal(n, okBytes.byteLength);
    assert.deepEqual(buf.subarray(0, n), okBytes);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('18. End-to-end real HTTP 206 server: simultaneous /boot/, /game/, /data/, cache isolation, and rpf:// mounted from /game/', async () => {
  const rpfSubFile = utf8Encoder.encode('<Settings><Mount>GameBase</Mount></Settings>');
  const rpfArchive = buildSyntheticRpf9Buffer(
    [{ path: 'config/settings.xml', data: rpfSubFile }],
    { magic: RPF9_MAGIC_LE, encryptionTag: RPF_ENCRYPTION_OPEN },
  );

  const bootManifest = utf8Encoder.encode('{"image":"main/image/guest.pgim"}');
  const pgimImage = makePatternBytes(384, 55);
  // Same relative filename `shared.json` under `/boot/` and `/data/` to verify cache key isolation!
  const bootShared = utf8Encoder.encode('{"source":"boot"}');
  const dataShared = utf8Encoder.encode('{"source":"data"}');

  const server = await startMultiBaseHttpServer(
    new Map([
      ['/boot/program-manifest.json', bootManifest],
      ['/boot/main/image/guest.pgim', pgimImage],
      ['/boot/shared.json', bootShared],
      ['/game/appdata0.rpf', rpfArchive],
      ['/data/shared.json', dataShared],
    ]),
  );

  const controller = new IoWorkerController({
    chunkSize: 256,
    maxBytes: 64 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(128);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({
      dataBase: `${server.origin}/data/`,
      bootBase: `${server.origin}/boot/`,
      gameBase: `${server.origin}/game/`,
    });
    await controller.attachMailbox(sab);

    // 1. /boot/program-manifest.json and /boot/main/image/guest.pgim
    const hBootManifest = await client.openAsync('boot/program-manifest.json');
    assert.ok(hBootManifest >= 1);
    const buf1 = new Uint8Array(128);
    const n1 = await client.readAsync(hBootManifest, 0n, buf1, 128);
    assert.deepEqual(buf1.subarray(0, n1), bootManifest);
    assert.equal(await client.closeAsync(hBootManifest), true);

    const hPgim = await client.openAsync('G:\\boot\\main\\image\\guest.pgim');
    assert.ok(hPgim >= 1);
    const buf2 = new Uint8Array(512);
    const n2 = await client.readAsync(hPgim, 0n, buf2, 512);
    assert.equal(n2, pgimImage.byteLength);
    assert.deepEqual(buf2.subarray(0, n2), pgimImage);
    assert.equal(await client.closeAsync(hPgim), true);

    // 2. Cache isolation: `boot/shared.json` vs `shared.json` return distinct payloads
    const hBootShared = await client.openAsync('boot/shared.json');
    const hDataShared = await client.openAsync('shared.json');
    assert.ok(hBootShared >= 1 && hDataShared >= 1);
    const outBootShared = new Uint8Array(64);
    const outDataShared = new Uint8Array(64);
    const lenBS = await client.readAsync(hBootShared, 0n, outBootShared, 64);
    const lenDS = await client.readAsync(hDataShared, 0n, outDataShared, 64);
    assert.equal(utf8Decoder.decode(outBootShared.subarray(0, lenBS)), '{"source":"boot"}');
    assert.equal(utf8Decoder.decode(outDataShared.subarray(0, lenDS)), '{"source":"data"}');
    assert.equal(await client.closeAsync(hBootShared), true);
    assert.equal(await client.closeAsync(hDataShared), true);

    // 3. Direct Xbox path `G:\appdata0.rpf` -> `/game/appdata0.rpf`
    const hRawRpf = await client.openAsync('G:\\appdata0.rpf');
    assert.ok(hRawRpf >= 1);
    const headerBuf = new Uint8Array(4);
    assert.equal(await client.readAsync(hRawRpf, 0n, headerBuf, 4), 4);
    assert.deepEqual(Array.from(headerBuf), [0x39, 0x46, 0x50, 0x52]); // LE "9FPR"
    assert.equal(await client.closeAsync(hRawRpf), true);

    // 4. Virtual `rpf://appdata0.rpf/config/settings.xml` lazily mounts from `/game/appdata0.rpf`!
    const hRpfSub = await client.openAsync('rpf://appdata0.rpf/config/settings.xml');
    assert.ok(hRpfSub >= 1);
    const subOut = new Uint8Array(128);
    const subLen = await client.readAsync(hRpfSub, 0n, subOut, 128);
    assert.equal(subLen, rpfSubFile.byteLength);
    assert.deepEqual(subOut.subarray(0, subLen), rpfSubFile);
    assert.equal(await client.closeAsync(hRpfSub), true);

    // Verify all 3 HTTP bases were hit with real HTTP 206 Range requests
    const rangedPaths = new Set(server.requests.filter((r) => r.rangeHeader).map((r) => r.pathname));
    assert.ok(rangedPaths.has('/boot/program-manifest.json'));
    assert.ok(rangedPaths.has('/boot/main/image/guest.pgim'));
    assert.ok(rangedPaths.has('/boot/shared.json'));
    assert.ok(rangedPaths.has('/data/shared.json'));
    assert.ok(rangedPaths.has('/game/appdata0.rpf'));
  } finally {
    await controller.detachMailbox();
    await server.close();
  }
});
