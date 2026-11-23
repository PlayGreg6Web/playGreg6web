'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
if (typeof importScripts === 'function') {
  importScripts('./workers/userdata.js');
}

const progress = new BroadcastChannel('game-progress');

const userdataApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6Userdata) || null;
const userdataStore = userdataApi ? new userdataApi.UserdataStore() : null;

let gpuWorker = null;
let ioWorker = null;
let audioWorker = null;
let inputSab = null;

function report(label) {
  progress.postMessage({ label });
}

function encodeTestFramePacket(r, g, b, a) {
  // Matches runtime/abi/playgta6_abi.h:
  // BeginFrame (8B) + ClearColor (8B + 16B) + Present (8B) = 40 bytes
  const buf = new ArrayBuffer(40);
  const view = new DataView(buf);
  // BeginFrame (opcode 1, payload 0)
  view.setUint16(0, 1, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 0, true);
  // ClearColor (opcode 2, payload 16)
  view.setUint16(8, 2, true);
  view.setUint16(10, 0, true);
  view.setUint32(12, 16, true);
  view.setFloat32(16, r, true);
  view.setFloat32(20, g, true);
  view.setFloat32(24, b, true);
  view.setFloat32(28, a, true);
  // Present (opcode 7, payload 0)
  view.setUint16(32, 7, true);
  view.setUint16(34, 0, true);
  view.setUint32(36, 0, true);
  return buf;
}

function createRuntimeModuleOptions() {
  const preRun = [
    (mod) => {
      if (!userdataStore || !mod || !mod.FS) return;
      const useRunDep =
        typeof mod.addRunDependency === 'function' &&
        typeof mod.removeRunDependency === 'function';
      if (useRunDep) {
        mod.addRunDependency('userdata-restore');
      }
      userdataStore
        .restoreAll(mod.FS, { timeoutMs: 8000 })
        .then((res) => {
          if (res.timedOut) {
            report(`userdata restore timed out (continuing boot): ${res.error}`);
          } else if (res.error) {
            report(`userdata restore skipped: ${res.error}`);
          } else {
            report(`userdata restored (${res.restored} files)`);
          }
        })
        .catch((err) => {
          report(`userdata restore warning: ${err}`);
        })
        .finally(() => {
          if (useRunDep) {
            mod.removeRunDependency('userdata-restore');
          }
        });
    },
  ];

  return {
    locateFile: (p) => `./dist/${p}`,
    preRun,
    hostCallbacks: {
      log: (msg) => report(`[wasm] ${msg}`),
      wgpuSubmit: (bytes) => {
        const copy = bytes.slice().buffer;
        gpuWorker.postMessage({ cmdBuffer: copy }, [copy]);
      },
      userdataRestore: (fsAdapter, opts) => {
        if (!userdataStore) {
          return Promise.resolve({
            restored: 0,
            skipped: 0,
            timedOut: false,
            error: 'userdata store unavailable',
          });
        }
        return userdataStore.restoreAll(fsAdapter, opts);
      },
      userdataSave: (path, bytes, mtime) => {
        if (!userdataStore) return Promise.resolve(null);
        return userdataStore.saveFile(path, bytes, mtime).catch((err) => {
          report(`userdata save warning (${path}): ${err}`);
          return null;
        });
      },
      userdataDelete: (path) => {
        if (!userdataStore) return Promise.resolve(null);
        return userdataStore.deleteFile(path).catch((err) => {
          report(`userdata delete warning (${path}): ${err}`);
          return null;
        });
      },
    },
  };
}

async function tryBootWasmRuntime() {
  try {
    const head = await fetch('./dist/runtime.js', { method: 'HEAD' });
    if (!head.ok) {
      report('No web/dist/runtime.js built yet (running WebGPU command-ring self-test).');
      const pkt = encodeTestFramePacket(0.08, 0.14, 0.24, 1.0);
      gpuWorker.postMessage({ cmdBuffer: pkt }, [pkt]);
      return;
    }
    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      await self.createRuntimeModule(createRuntimeModuleOptions());
    }
  } catch (err) {
    progress.postMessage({ error: `runtime boot: ${err}` });
  }
}

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.resize && gpuWorker) {
    gpuWorker.postMessage({ resize: true, width: msg.width, height: msg.height });
    return;
  }

  if (!msg.canvas) return;
  inputSab = msg.inputSab || null;

  ioWorker = new Worker('./workers/io_worker.js');
  gpuWorker = new Worker('./workers/wgpu_worker.js');
  audioWorker = new Worker('./workers/audio_worker.js');

  let readyCount = 0;
  const onWorkerReady = (name, detail) => {
    report(`${name} ready${detail ? ` (${detail})` : ''}`);
    if (++readyCount === 3) {
      tryBootWasmRuntime();
    }
  };

  ioWorker.onmessage = (e) => {
    if (e.data.ready) onWorkerReady('io_worker', e.data.summary);
    else if (e.data.error) progress.postMessage({ error: `io_worker: ${e.data.error}` });
  };
  gpuWorker.onmessage = (e) => {
    if (e.data.ready) onWorkerReady('wgpu_worker', e.data.adapter);
    else if (e.data.frame) report(`WebGPU frame #${e.data.frame} presented`);
    else if (e.data.error) progress.postMessage({ error: `wgpu_worker: ${e.data.error}` });
  };
  audioWorker.onmessage = (e) => {
    if (e.data.ready) onWorkerReady('audio_worker');
  };

  ioWorker.postMessage({ init: true, dataBase: msg.dataBase });
  gpuWorker.postMessage(
    { init: true, canvas: msg.canvas, width: msg.width, height: msg.height },
    [msg.canvas]
  );
  audioWorker.postMessage({ init: true });
};
