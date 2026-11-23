'use strict';

// IndexedDB persistence for Wasm /userdata/ saves and settings.
// Enforces strict /userdata/ path confinement and safe restore timeouts so a
// broken or hung IndexedDB never blocks runtime boot.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_DB_NAME = 'playgta6-userdata';
const DEFAULT_STORE_NAME = 'files';
const DEFAULT_RESTORE_TIMEOUT_MS = 8000;
const USERDATA_PREFIX = '/userdata/';

function validateUserdataPath(path) {
  if (typeof path !== 'string' || path.length === 0) {
    throw new Error('Userdata path must be a non-empty string');
  }
  if (path.includes('\\') || path.includes('\0')) {
    throw new Error(`Invalid userdata path (forbidden character): ${path}`);
  }
  if (!path.startsWith(USERDATA_PREFIX)) {
    throw new Error(`Userdata path must start with ${USERDATA_PREFIX}: ${path}`);
  }
  if (path.length === USERDATA_PREFIX.length || path.endsWith('/')) {
    throw new Error(`Userdata path must reference a file inside ${USERDATA_PREFIX}: ${path}`);
  }

  const rel = path.slice(USERDATA_PREFIX.length);
  const segments = rel.split('/');
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..') {
      throw new Error(`Invalid userdata path segment in: ${path}`);
    }
  }
  return path;
}

function toUint8Array(bytes) {
  if (bytes instanceof Uint8Array) {
    return bytes.slice();
  }
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice();
  }
  if (bytes instanceof ArrayBuffer) {
    return new Uint8Array(bytes.slice(0));
  }
  if (typeof bytes === 'string') {
    return new TextEncoder().encode(bytes);
  }
  throw new TypeError('Userdata file payload must be a Uint8Array, ArrayBufferView, ArrayBuffer, or string');
}

async function ensureParentDirs(fsAdapter, filePath) {
  if (!fsAdapter) return;
  const lastSlash = filePath.lastIndexOf('/');
  if (lastSlash <= 0) return;
  const parentDir = filePath.slice(0, lastSlash);

  if (typeof fsAdapter.mkdirTree === 'function') {
    await fsAdapter.mkdirTree(parentDir);
    return;
  }

  if (typeof fsAdapter.mkdir === 'function') {
    const parts = parentDir.split('/').filter(Boolean);
    let current = '';
    for (const part of parts) {
      current += `/${part}`;
      try {
        await fsAdapter.mkdir(current);
      } catch {
        // Ignore already-existing directory errors (e.g. Emscripten FS EEXIST).
      }
    }
  }
}

class UserdataStore {
  constructor(options = {}) {
    this.dbName = typeof options.dbName === 'string' && options.dbName ? options.dbName : DEFAULT_DB_NAME;
    this.storeName =
      typeof options.storeName === 'string' && options.storeName ? options.storeName : DEFAULT_STORE_NAME;
    this.dbVersion =
      Number.isInteger(options.dbVersion) && options.dbVersion > 0 ? options.dbVersion : 1;
    this.indexedDB =
      options.indexedDB !== undefined
        ? options.indexedDB
        : typeof globalThis !== 'undefined' && globalThis.indexedDB
          ? globalThis.indexedDB
          : null;
    this.backend = options.backend || null;

    this._db = null;
    this._dbPromise = null;
  }

  _openDb() {
    if (this._db) {
      return Promise.resolve(this._db);
    }
    if (this._dbPromise) {
      return this._dbPromise;
    }
    if (!this.indexedDB || typeof this.indexedDB.open !== 'function') {
      return Promise.resolve(null);
    }

    this._dbPromise = new Promise((resolve, reject) => {
      let req;
      try {
        req = this.indexedDB.open(this.dbName, this.dbVersion);
      } catch (err) {
        this._dbPromise = null;
        reject(err);
        return;
      }

      req.onupgradeneeded = () => {
        const db = req.result;
        const hasStore =
          db.objectStoreNames &&
          (typeof db.objectStoreNames.contains === 'function'
            ? db.objectStoreNames.contains(this.storeName)
            : Array.from(db.objectStoreNames).includes(this.storeName));
        if (!hasStore) {
          db.createObjectStore(this.storeName, { keyPath: 'path' });
        }
      };

      req.onsuccess = () => {
        const db = req.result;
        this._db = db;
        if (db && typeof db === 'object') {
          db.onversionchange = () => {
            try {
              db.close();
            } catch {
              // Ignore close errors during version change.
            }
            if (this._db === db) {
              this._db = null;
              this._dbPromise = null;
            }
          };
          db.onclose = () => {
            if (this._db === db) {
              this._db = null;
              this._dbPromise = null;
            }
          };
        }
        resolve(db);
      };

      req.onerror = () => {
        this._dbPromise = null;
        reject(req.error || new Error(`Failed to open IndexedDB database "${this.dbName}"`));
      };
    });

    return this._dbPromise;
  }

  async _runTransaction(mode, operation) {
    const db = await this._openDb();
    if (!db) {
      throw new Error('IndexedDB is unavailable');
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      const finishResolve = (value) => {
        if (!settled) {
          settled = true;
          resolve(value);
        }
      };
      const finishReject = (err) => {
        if (!settled) {
          settled = true;
          reject(err);
        }
      };

      let tx;
      let req;
      try {
        tx = db.transaction(this.storeName, mode);
        const store = tx.objectStore(this.storeName);
        req = operation(store);
      } catch (err) {
        finishReject(err);
        return;
      }

      let reqResult;
      let reqDone = false;
      const waitForTxComplete = tx && 'oncomplete' in tx;

      req.onsuccess = () => {
        reqResult = req.result;
        reqDone = true;
        if (!waitForTxComplete) {
          finishResolve(reqResult);
        }
      };

      req.onerror = () => {
        finishReject(req.error || new Error(`IndexedDB request failed (${mode})`));
      };

      if (tx) {
        tx.oncomplete = () => {
          finishResolve(reqDone ? reqResult : req.result);
        };
        tx.onerror = () => {
          finishReject(tx.error || req.error || new Error(`IndexedDB transaction failed (${mode})`));
        };
        tx.onabort = () => {
          finishReject(tx.error || new Error(`IndexedDB transaction aborted (${mode})`));
        };
      }
    });
  }

  async _getAllRecords() {
    if (this.backend) {
      const records = await this.backend.getAll();
      return Array.isArray(records) ? records : [];
    }
    const records = await this._runTransaction('readonly', (store) => store.getAll());
    return Array.isArray(records) ? records : [];
  }

  async restoreAll(fsAdapter, options = {}) {
    const timeoutMs =
      Number.isFinite(options.timeoutMs) && options.timeoutMs > 0
        ? options.timeoutMs
        : DEFAULT_RESTORE_TIMEOUT_MS;

    if (!fsAdapter || typeof fsAdapter.writeFile !== 'function') {
      throw new TypeError('restoreAll requires an fsAdapter with a writeFile(path, bytes) method');
    }

    if (!this.backend && (!this.indexedDB || typeof this.indexedDB.open !== 'function')) {
      return { restored: 0, skipped: 0, timedOut: false, error: 'IndexedDB unavailable' };
    }

    let timer = null;
    let timedOut = false;

    const timeoutPromise = new Promise((resolve) => {
      timer = setTimeout(() => {
        timedOut = true;
        if (!this._db) {
          this._dbPromise = null;
        }
        resolve({
          restored: 0,
          skipped: 0,
          timedOut: true,
          error: `IndexedDB restore timed out after ${timeoutMs}ms`,
        });
      }, timeoutMs);
    });

    const restorePromise = (async () => {
      let restored = 0;
      let skipped = 0;
      try {
        const records = await this._getAllRecords();
        if (timedOut) {
          return {
            restored: 0,
            skipped: 0,
            timedOut: true,
            error: `IndexedDB restore timed out after ${timeoutMs}ms`,
          };
        }

        if (typeof fsAdapter.mkdirTree === 'function') {
          await fsAdapter.mkdirTree('/userdata');
        } else if (typeof fsAdapter.mkdir === 'function') {
          try {
            await fsAdapter.mkdir('/userdata');
          } catch {
            // Ignore already-existing /userdata directory.
          }
        }

        for (const rec of records) {
          if (timedOut) break;
          if (!rec || typeof rec.path !== 'string') {
            skipped += 1;
            continue;
          }
          let validPath;
          try {
            validPath = validateUserdataPath(rec.path);
          } catch {
            skipped += 1;
            continue;
          }

          let payload;
          try {
            payload = toUint8Array(rec.data);
          } catch {
            skipped += 1;
            continue;
          }

          await ensureParentDirs(fsAdapter, validPath);
          await fsAdapter.writeFile(validPath, payload, rec.mtime);
          if (typeof fsAdapter.utime === 'function' && Number.isFinite(rec.mtime)) {
            try {
              await fsAdapter.utime(validPath, rec.mtime, rec.mtime);
            } catch {
              // Setting mtime is best-effort.
            }
          }
          restored += 1;
        }

        return { restored, skipped, timedOut: false, error: null };
      } catch (err) {
        return { restored, skipped, timedOut: false, error: String(err) };
      }
    })();

    try {
      return await Promise.race([restorePromise, timeoutPromise]);
    } finally {
      if (timer !== null) {
        clearTimeout(timer);
      }
    }
  }

  async saveFile(path, bytes, mtime = Date.now()) {
    const validPath = validateUserdataPath(path);
    const data = toUint8Array(bytes);
    const ts = Number.isFinite(mtime) ? mtime : Date.now();
    const record = { path: validPath, data, mtime: ts };

    if (this.backend) {
      await this.backend.put(record);
    } else {
      await this._runTransaction('readwrite', (store) => store.put(record));
    }

    return { path: validPath, size: data.byteLength, mtime: ts };
  }

  async deleteFile(path) {
    const validPath = validateUserdataPath(path);

    if (this.backend) {
      await this.backend.delete(validPath);
    } else {
      await this._runTransaction('readwrite', (store) => store.delete(validPath));
    }

    return { path: validPath, deleted: true };
  }

  async listFiles() {
    const records = await this._getAllRecords();
    const out = [];
    for (const rec of records) {
      if (!rec || typeof rec.path !== 'string') continue;
      try {
        validateUserdataPath(rec.path);
        out.push({
          path: rec.path,
          data: toUint8Array(rec.data),
          mtime: Number.isFinite(rec.mtime) ? rec.mtime : 0,
        });
      } catch {
        // Skip invalid entries.
      }
    }
    return out;
  }

  close() {
    if (this._db && typeof this._db.close === 'function') {
      try {
        this._db.close();
      } catch {
        // Ignore close errors.
      }
    }
    this._db = null;
    this._dbPromise = null;
    if (this.backend && typeof this.backend.close === 'function') {
      this.backend.close();
    }
  }
}

const exportedApi = {
  UserdataStore,
  DEFAULT_DB_NAME,
  DEFAULT_STORE_NAME,
  DEFAULT_RESTORE_TIMEOUT_MS,
  USERDATA_PREFIX,
  validateUserdataPath,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Userdata = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
