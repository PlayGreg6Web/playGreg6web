// Interactive browser HUD diagnostics & controls controller for playgta6web.
// Binds HUD actions (OPFS cache clear, userdata save backup/restore, GPU .pg6trace
// recording/export, sanitized crash report export, input profile switching, and
// smoke demo / runtime session controls) cleanly to BrowserStorageManager,
// GpuFrameRecorder / SmokeDemoRunner, CrashReporter, InputProfileManager,
// PerfMonitor, and RuntimeSessionController.
// Zero external dependencies; ES module compatible with Browser and Node.js.

import { formatBytes, SNAPSHOT_FORMAT } from '../storage/storage_manager.js';
import { SESSION_STATE } from '../runtime/session_controller.js';

export const HUD_ACTIONS = Object.freeze({
  CLEAR_OPFS_CACHE: 'clear_opfs_cache',
  EXPORT_SAVES: 'export_saves',
  IMPORT_SAVES: 'import_saves',
  TOGGLE_GPU_RECORDING: 'toggle_gpu_recording',
  EXPORT_GPU_TRACE: 'export_gpu_trace',
  EXPORT_CRASH_REPORT: 'export_crash_report',
  EXPORT_MISSING_SHADERS: 'export_missing_shaders',
  SWITCH_INPUT_PROFILE: 'switch_input_profile',
  TOGGLE_SMOKE_DEMO: 'toggle_smoke_demo',
  RESTART_SESSION: 'restart_session',
});

export const DEFAULT_FILENAMES = Object.freeze({
  USERDATA_BACKUP: 'playgta6-userdata-backup.json',
  GPU_TRACE: 'playgta6-capture.pg6trace',
  CRASH_REPORT: 'playgta6-diagnostic-report.json',
  MISSING_SHADERS: 'playgta6-missing-shaders.json',
});

const SUPPORTED_BUTTON_BINDINGS = Object.freeze([
  ['clearCacheBtn', 'click'],
  ['exportSavesBtn', 'click'],
  ['recordTraceBtn', 'click'],
  ['exportTraceBtn', 'click'],
  ['exportCrashBtn', 'click'],
  ['exportMissingShadersBtn', 'click'],
  ['restartBtn', 'click'],
  ['smokeDemoBtn', 'click'],
  ['profileSelect', 'change'],
]);

function toErrorMessage(err, fallback = 'Unknown error') {
  if (err instanceof Error && typeof err.message === 'string' && err.message.length > 0) {
    return err.message;
  }
  if (typeof err === 'string' && err.length > 0) {
    return err;
  }
  if (err && typeof err === 'object' && typeof err.message === 'string' && err.message.length > 0) {
    return err.message;
  }
  return fallback;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  throw new TypeError('Expected binary trace data (Uint8Array, ArrayBufferView, or ArrayBuffer)');
}

function createBrowserDownloadBlob() {
  if (
    typeof document === 'undefined' ||
    !document ||
    typeof document.createElement !== 'function' ||
    typeof Blob !== 'function' ||
    typeof URL === 'undefined' ||
    !URL ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return null;
  }

  return function defaultBrowserDownloadBlob(filename, data, mimeType = 'application/octet-stream') {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    try {
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      anchor.style.display = 'none';
      if (document.body && typeof document.body.appendChild === 'function') {
        document.body.appendChild(anchor);
      }
      anchor.click();
      if (anchor.parentNode && typeof anchor.parentNode.removeChild === 'function') {
        anchor.parentNode.removeChild(anchor);
      }
    } finally {
      if (typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(url);
      }
    }
  };
}

function createBrowserCopyToClipboard() {
  if (
    typeof navigator !== 'undefined' &&
    navigator &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === 'function'
  ) {
    return (text) => navigator.clipboard.writeText(String(text));
  }
  return null;
}

function validateEventTarget(el, label) {
  if (
    !el ||
    typeof el !== 'object' ||
    typeof el.addEventListener !== 'function' ||
    typeof el.removeEventListener !== 'function'
  ) {
    throw new TypeError(
      `bindButtons("${label}") requires an EventTarget with addEventListener/removeEventListener`
    );
  }
}

export class HudController {
  /**
   * @param {object} [options]
   * @param {object|null} [options.hud=null]
   * @param {object|null} [options.app=null]
   * @param {object|null} [options.sessionController=null]
   * @param {object|null} [options.smokeDemoRunner=null]
   * @param {object|null} [options.storageManager=null]
   * @param {object|null} [options.gpuRecorder=null]
   * @param {object|null} [options.gpuBootPanel=null]
   * @param {object|null} [options.crashReporter=null]
   * @param {object|null} [options.inputProfiles=null]
   * @param {object|null} [options.perfMonitor=null]
   * @param {((filename: string, data: string|Uint8Array, mimeType: string) => any)|null} [options.downloadBlob=null]
   * @param {((text: string) => any)|null} [options.copyToClipboard=null]
   * @param {() => number} [options.now]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('HudController options must be an object');
    }

    const {
      hud = null,
      app = null,
      sessionController = null,
      smokeDemoRunner = null,
      storageManager = null,
      gpuRecorder = null,
      gpuBootPanel = null,
      crashReporter = null,
      inputProfiles = null,
      perfMonitor = null,
      downloadBlob = undefined,
      copyToClipboard = undefined,
      now = () => Date.now(),
    } = options;

    if (downloadBlob !== undefined && downloadBlob !== null && typeof downloadBlob !== 'function') {
      throw new TypeError('downloadBlob must be a function or null');
    }
    if (
      copyToClipboard !== undefined &&
      copyToClipboard !== null &&
      typeof copyToClipboard !== 'function'
    ) {
      throw new TypeError('copyToClipboard must be a function or null');
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning epoch milliseconds');
    }

    this._hud = hud || null;
    this._app = app || null;
    this._sessionController = sessionController || null;
    this._smokeDemoRunner = smokeDemoRunner || null;
    this._storageManager = storageManager || null;
    this._gpuRecorder = gpuRecorder || null;
    this._gpuBootPanel = gpuBootPanel || (app && app.gpuBootPanel) || null;
    this._crashReporter = crashReporter || null;
    this._inputProfiles = inputProfiles || null;
    this._perfMonitor = perfMonitor || null;
    this._downloadBlob =
      downloadBlob === undefined ? createBrowserDownloadBlob() : downloadBlob;
    this._copyToClipboard =
      copyToClipboard === undefined ? createBrowserCopyToClipboard() : copyToClipboard;
    this._now = now;

    this._lastAction = null;
    this._boundListeners = [];
    this._boundElements = {};
    this._exportingMissingShaders = false;
    this._destroyed = false;
  }

  get isExportingMissingShaders() {
    return this._exportingMissingShaders;
  }

  // --- Internal Helpers ---

  _getGpuBootPanel() {
    if (this._gpuBootPanel) {
      return this._gpuBootPanel;
    }
    if (this._app && this._app.gpuBootPanel) {
      return this._app.gpuBootPanel;
    }
    return null;
  }

  _hudLog(message, level = 'info') {
    if (this._hud && typeof this._hud.appendLog === 'function') {
      try {
        this._hud.appendLog(String(message ?? ''), level);
      } catch {
        // Never allow HUD logging errors to break action execution.
      }
    }
  }

  _finishAction(action, ok, summary, extras = {}, logLevel = ok ? 'ok' : 'error') {
    const error = ok ? undefined : extras.error || summary;
    const timestampMs = this._now();

    this._lastAction = {
      ok: Boolean(ok),
      action,
      summary,
      error: ok ? null : error,
      timestampMs,
    };

    this._hudLog(summary, logLevel);

    const result = {
      ok: Boolean(ok),
      action,
      summary,
      ...extras,
    };
    if (!ok) {
      result.error = error;
    }
    return result;
  }

  async _triggerDownload(filename, data, mimeType) {
    if (typeof this._downloadBlob !== 'function') {
      return false;
    }
    await this._downloadBlob(filename, data, mimeType);
    return true;
  }

  async _triggerClipboardCopy(text) {
    if (typeof this._copyToClipboard !== 'function') {
      return false;
    }
    await this._copyToClipboard(text);
    return true;
  }

  _isRecording() {
    try {
      if (this._gpuRecorder && typeof this._gpuRecorder.isRecording === 'function') {
        return Boolean(this._gpuRecorder.isRecording());
      }
      if (
        this._smokeDemoRunner &&
        this._smokeDemoRunner.recorder &&
        typeof this._smokeDemoRunner.recorder.isRecording === 'function'
      ) {
        return Boolean(this._smokeDemoRunner.recorder.isRecording());
      }
    } catch {
      return false;
    }
    return false;
  }

  _getSessionState() {
    try {
      if (this._sessionController && typeof this._sessionController.getState === 'function') {
        return String(this._sessionController.getState());
      }
      if (this._smokeDemoRunner) {
        return this._smokeDemoRunner.running ? SESSION_STATE.RUNNING_SMOKE : SESSION_STATE.IDLE;
      }
    } catch {
      return SESSION_STATE.ERROR;
    }
    return SESSION_STATE.IDLE;
  }

  _getSessionMode() {
    try {
      if (this._sessionController && typeof this._sessionController.getMode === 'function') {
        return String(this._sessionController.getMode());
      }
      if (this._smokeDemoRunner && this._smokeDemoRunner.running) {
        return 'smoke';
      }
    } catch {
      return 'none';
    }
    return 'none';
  }

  _getActiveProfileId() {
    try {
      if (this._inputProfiles && typeof this._inputProfiles.getActiveProfile === 'function') {
        const active = this._inputProfiles.getActiveProfile();
        return active && typeof active.id === 'string' ? active.id : null;
      }
    } catch {
      return null;
    }
    return null;
  }

  _syncBoundControlState() {
    const recordBtn = this._boundElements.recordTraceBtn;
    if (recordBtn) {
      const recording = this._isRecording();
      try {
        if (recordBtn.dataset && typeof recordBtn.dataset === 'object') {
          recordBtn.dataset.recording = recording ? 'true' : 'false';
        }
        if (typeof recordBtn.setAttribute === 'function') {
          recordBtn.setAttribute('aria-pressed', recording ? 'true' : 'false');
        }
      } catch {
        // Ignore DOM attribute update errors on lightweight mocks.
      }
    }

    const profileSelect = this._boundElements.profileSelect;
    if (profileSelect) {
      const activeId = this._getActiveProfileId();
      if (activeId !== null) {
        try {
          profileSelect.value = activeId;
        } catch {
          // Ignore read-only mock properties.
        }
      }
    }
  }

  // --- Interactive Actions ---

  /**
   * Clears the OPFS range cache via `storageManager.clearOpfsCache()` and logs
   * removed entry count + freed bytes to the HUD.
   */
  async clearOpfsCache() {
    const action = HUD_ACTIONS.CLEAR_OPFS_CACHE;
    if (!this._storageManager || typeof this._storageManager.clearOpfsCache !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot clear OPFS cache.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      const result = await this._storageManager.clearOpfsCache();
      if (!result || !result.supported) {
        return this._finishAction(
          action,
          false,
          'OPFS cache is unavailable in this browser.',
          { error: 'OPFS cache is unsupported', result: result ?? null },
          'warn'
        );
      }

      const removedCount = Number.isFinite(result.removedCount) ? result.removedCount : 0;
      const freedBytes = Number.isFinite(result.freedBytes) ? result.freedBytes : 0;
      const freedFormatted = formatBytes(freedBytes);
      const summary = `Cleared OPFS cache: removed ${removedCount} file${removedCount === 1 ? '' : 's'}, freed ${freedFormatted} (${freedBytes} B).`;

      return this._finishAction(
        action,
        true,
        summary,
        { removedCount, freedBytes, freedFormatted, result },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to clear OPFS cache');
      return this._finishAction(action, false, `Failed to clear OPFS cache: ${error}`, { error });
    }
  }

  /**
   * Exports `/userdata/` saves via `storageManager.exportUserdataSnapshot()`, triggers a
   * JSON download (`playgta6-userdata-backup.json`) if `downloadBlob` is available,
   * and returns the serialized snapshot JSON.
   */
  async exportSaves() {
    const action = HUD_ACTIONS.EXPORT_SAVES;
    if (!this._storageManager || typeof this._storageManager.exportUserdataSnapshot !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot export saves.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      const snapshot = await this._storageManager.exportUserdataSnapshot();
      if (!snapshot || !snapshot.supported) {
        return this._finishAction(
          action,
          false,
          'IndexedDB userdata storage is unavailable; cannot export saves.',
          { error: 'IndexedDB userdata storage is unsupported', snapshot: snapshot ?? null },
          'warn'
        );
      }

      const json = JSON.stringify(snapshot, null, 2);
      const filename = DEFAULT_FILENAMES.USERDATA_BACKUP;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');

      const fileCount = Number.isFinite(snapshot.fileCount) ? snapshot.fileCount : 0;
      const totalBytes = Number.isFinite(snapshot.totalBytes) ? snapshot.totalBytes : 0;
      const summary = `Exported ${fileCount} save file${fileCount === 1 ? '' : 's'} (${formatBytes(totalBytes)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, snapshot, fileCount, totalBytes, downloaded },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export saves');
      return this._finishAction(action, false, `Failed to export saves: ${error}`, { error });
    }
  }

  /**
   * Parses and validates a userdata backup JSON string (or snapshot object) and restores
   * it via `storageManager.importUserdataSnapshot(parsed, { overwrite })`.
   *
   * @param {string|object} jsonStr
   * @param {{ overwrite?: boolean }} [options]
   */
  async importSaves(jsonStr, options = {}) {
    const action = HUD_ACTIONS.IMPORT_SAVES;
    const overwrite = options && typeof options === 'object' && options.overwrite !== undefined
      ? Boolean(options.overwrite)
      : true;

    if (!this._storageManager || typeof this._storageManager.importUserdataSnapshot !== 'function') {
      return this._finishAction(
        action,
        false,
        'Storage manager unavailable; cannot import saves.',
        { error: 'Storage manager is not configured' },
        'warn'
      );
    }

    try {
      let parsed;
      if (typeof jsonStr === 'string') {
        if (jsonStr.trim().length === 0) {
          throw new SyntaxError('Save backup JSON string is empty');
        }
        parsed = JSON.parse(jsonStr);
      } else if (isPlainObject(jsonStr)) {
        parsed = jsonStr;
      } else {
        throw new TypeError('Save backup must be a JSON string or snapshot object');
      }

      if (!isPlainObject(parsed)) {
        throw new TypeError('Save backup JSON root must be an object');
      }
      if (parsed.format !== undefined && parsed.format !== SNAPSHOT_FORMAT) {
        throw new Error(`Unsupported save backup format: ${String(parsed.format)}`);
      }
      if (!Array.isArray(parsed.files)) {
        throw new TypeError('Save backup JSON must include a files array');
      }

      const result = await this._storageManager.importUserdataSnapshot(parsed, { overwrite });
      if (!result || !result.supported) {
        return this._finishAction(
          action,
          false,
          'IndexedDB userdata storage is unavailable; cannot import saves.',
          { error: 'IndexedDB userdata storage is unsupported', result: result ?? null },
          'warn'
        );
      }

      const importedCount = Number.isFinite(result.importedCount) ? result.importedCount : 0;
      const skippedCount = Number.isFinite(result.skippedCount) ? result.skippedCount : 0;
      const importedBytes = Number.isFinite(result.importedBytes) ? result.importedBytes : 0;
      const summary = `Imported ${importedCount} save file${importedCount === 1 ? '' : 's'} (${formatBytes(importedBytes)}, skipped ${skippedCount}).`;

      return this._finishAction(
        action,
        true,
        summary,
        { importedCount, skippedCount, importedBytes, overwrite, result },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to import saves');
      return this._finishAction(action, false, `Failed to import saves: ${error}`, { error });
    }
  }

  /**
   * Toggles GPU frame recording on `gpuRecorder` or `smokeDemoRunner`.
   */
  async toggleGpuRecording() {
    const action = HUD_ACTIONS.TOGGLE_GPU_RECORDING;
    const hasGpuRecorder =
      this._gpuRecorder &&
      typeof this._gpuRecorder.start === 'function' &&
      typeof this._gpuRecorder.stop === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner &&
      typeof this._smokeDemoRunner.startRecording === 'function' &&
      typeof this._smokeDemoRunner.stopRecording === 'function';

    if (!hasGpuRecorder && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'No GPU frame recorder or smoke demo runner configured.',
        { error: 'GPU recorder is not configured', isRecording: false },
        'warn'
      );
    }

    try {
      const wasRecording = this._isRecording();
      let metadata = null;

      if (hasGpuRecorder) {
        if (wasRecording) {
          this._gpuRecorder.stop();
        } else {
          this._gpuRecorder.start();
        }
        if (typeof this._gpuRecorder.exportMetadata === 'function') {
          metadata = this._gpuRecorder.exportMetadata();
        }
      } else {
        if (wasRecording) {
          this._smokeDemoRunner.stopRecording();
        } else {
          this._smokeDemoRunner.startRecording();
        }
        if (typeof this._smokeDemoRunner.exportTraceMetadata === 'function') {
          metadata = this._smokeDemoRunner.exportTraceMetadata();
        }
      }

      const isRecording = this._isRecording();
      this._syncBoundControlState();

      const frameCount = metadata && Number.isFinite(metadata.frameCount) ? metadata.frameCount : 0;
      const summary = isRecording
        ? 'GPU trace recording started.'
        : `GPU trace recording stopped (${frameCount} frame${frameCount === 1 ? '' : 's'} buffered).`;

      return this._finishAction(
        action,
        true,
        summary,
        { isRecording, metadata },
        isRecording ? 'info' : 'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to toggle GPU recording');
      return this._finishAction(action, false, `Failed to toggle GPU recording: ${error}`, {
        error,
        isRecording: this._isRecording(),
      });
    }
  }

  /**
   * Exports buffered GPU frames as a `.pg6trace` `Uint8Array` via
   * `downloadBlob('playgta6-capture.pg6trace', bytes, 'application/octet-stream')`.
   *
   * @param {{ frameStart?: number, frameCount?: number }} [options]
   */
  async exportGpuTrace(options) {
    const action = HUD_ACTIONS.EXPORT_GPU_TRACE;
    const hasGpuRecorder =
      this._gpuRecorder && typeof this._gpuRecorder.exportTraceBytes === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner && typeof this._smokeDemoRunner.exportTraceBytes === 'function';

    if (!hasGpuRecorder && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'No GPU frame recorder or smoke demo runner configured.',
        { error: 'GPU recorder is not configured' },
        'warn'
      );
    }

    try {
      let rawBytes;
      let metadata = null;

      if (hasGpuRecorder) {
        rawBytes = this._gpuRecorder.exportTraceBytes(options);
        if (typeof this._gpuRecorder.exportMetadata === 'function') {
          metadata = this._gpuRecorder.exportMetadata();
        }
      } else {
        rawBytes = this._smokeDemoRunner.exportTraceBytes(options);
        if (typeof this._smokeDemoRunner.exportTraceMetadata === 'function') {
          metadata = this._smokeDemoRunner.exportTraceMetadata();
        }
      }

      const bytes = toUint8Array(rawBytes);
      if (bytes.byteLength === 0) {
        return this._finishAction(
          action,
          false,
          'No recorded GPU frames available to export.',
          {
            error: 'No recorded GPU frames available',
            bytes,
            byteLength: 0,
            frameCount: 0,
            metadata,
            downloaded: false,
          },
          'warn'
        );
      }

      const filename = DEFAULT_FILENAMES.GPU_TRACE;
      const downloaded = await this._triggerDownload(filename, bytes, 'application/octet-stream');
      const frameCount = metadata && Number.isFinite(metadata.frameCount) ? metadata.frameCount : null;
      const frameDetail = frameCount !== null ? ` (${frameCount} frame${frameCount === 1 ? '' : 's'})` : '';
      const summary = `Exported GPU trace${frameDetail} (${formatBytes(bytes.byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        {
          filename,
          bytes,
          byteLength: bytes.byteLength,
          frameCount,
          metadata,
          downloaded,
        },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export GPU trace');
      return this._finishAction(action, false, `Failed to export GPU trace: ${error}`, { error });
    }
  }

  /**
   * Builds a sanitized diagnostic report via `crashReporter.toJson({ perfSnapshot, storageSummary, serverStatus })`
   * and triggers a JSON download (`playgta6-diagnostic-report.json`).
   *
   * @param {{ perfSnapshot?: object, storageSummary?: object, serverStatus?: object, extra?: unknown, copyToClipboard?: boolean }} [options]
   */
  async exportCrashReport(options = {}) {
    const action = HUD_ACTIONS.EXPORT_CRASH_REPORT;
    if (!this._crashReporter || typeof this._crashReporter.toJson !== 'function') {
      return this._finishAction(
        action,
        false,
        'Crash reporter unavailable; cannot export diagnostic report.',
        { error: 'Crash reporter is not configured' },
        'warn'
      );
    }

    try {
      const opts = isPlainObject(options) ? options : {};

      let sessionDiagnostics = null;
      if (
        this._sessionController &&
        typeof this._sessionController.getDiagnostics === 'function'
      ) {
        try {
          sessionDiagnostics = this._sessionController.getDiagnostics();
        } catch {
          sessionDiagnostics = null;
        }
      }

      let perfSnapshot = opts.perfSnapshot;
      if (perfSnapshot === undefined) {
        if (this._perfMonitor && typeof this._perfMonitor.getSnapshot === 'function') {
          try {
            perfSnapshot = this._perfMonitor.getSnapshot();
          } catch {
            perfSnapshot = null;
          }
        } else if (sessionDiagnostics && sessionDiagnostics.perfSnapshot !== undefined) {
          perfSnapshot = sessionDiagnostics.perfSnapshot;
        }
      }

      let storageSummary = opts.storageSummary;
      if (storageSummary === undefined) {
        if (this._storageManager && typeof this._storageManager.getSummary === 'function') {
          try {
            storageSummary = await this._storageManager.getSummary();
          } catch {
            storageSummary = sessionDiagnostics ? sessionDiagnostics.storageSummary : null;
          }
        } else if (sessionDiagnostics && sessionDiagnostics.storageSummary !== undefined) {
          storageSummary = sessionDiagnostics.storageSummary;
        }
      }

      let serverStatus = opts.serverStatus;
      if (serverStatus === undefined && sessionDiagnostics && sessionDiagnostics.serverStatus !== undefined) {
        serverStatus = sessionDiagnostics.serverStatus;
      }

      const reportInput = {
        perfSnapshot,
        storageSummary,
        serverStatus,
      };
      if (opts.extra !== undefined) {
        reportInput.extra = opts.extra;
      }

      const json = this._crashReporter.toJson(reportInput);
      const filename = DEFAULT_FILENAMES.CRASH_REPORT;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');
      const copied = opts.copyToClipboard === true
        ? await this._triggerClipboardCopy(json)
        : false;
      const byteLength = new TextEncoder().encode(json).byteLength;
      const summary = `Exported sanitized diagnostic report (${formatBytes(byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, byteLength, downloaded, copied },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export diagnostic report');
      return this._finishAction(
        action,
        false,
        `Failed to export diagnostic report: ${error}`,
        { error }
      );
    }
  }

  /**
   * Exports the sanitized missing-shader shopping list JSON via
   * `gpuBootPanel.exportMissingShadersJson()` and triggers a JSON download
   * (`playgta6-missing-shaders.json`) when `downloadBlob` is available.
   *
   * @param {{ copyToClipboard?: boolean }} [options]
   */
  async exportMissingShaders(options = {}) {
    const action = HUD_ACTIONS.EXPORT_MISSING_SHADERS;
    const panel = this._getGpuBootPanel();
    if (!panel || typeof panel.exportMissingShadersJson !== 'function') {
      return this._finishAction(
        action,
        false,
        'GPU boot panel unavailable; cannot export missing shaders.',
        { error: 'GPU boot panel is not configured' },
        'warn'
      );
    }

    try {
      const opts = isPlainObject(options) ? options : {};
      let json;
      this._exportingMissingShaders = true;
      try {
        json = panel.exportMissingShadersJson();
      } finally {
        this._exportingMissingShaders = false;
      }

      if (typeof json !== 'string' || json.length === 0) {
        throw new TypeError('GPU boot panel did not return a missing-shader JSON string');
      }

      let missingShaderCount = null;
      try {
        const parsed = JSON.parse(json);
        if (parsed && typeof parsed === 'object') {
          if (Number.isFinite(parsed.missingShaderCount)) {
            missingShaderCount = parsed.missingShaderCount;
          } else if (Array.isArray(parsed.missingShaders)) {
            missingShaderCount = parsed.missingShaders.length;
          }
        }
      } catch {
        // Leave missingShaderCount null if a custom mock returned non-JSON text.
      }

      const filename = DEFAULT_FILENAMES.MISSING_SHADERS;
      const downloaded = await this._triggerDownload(filename, json, 'application/json');
      const copied =
        opts.copyToClipboard === true ? await this._triggerClipboardCopy(json) : false;
      const byteLength = new TextEncoder().encode(json).byteLength;
      const countDetail =
        missingShaderCount !== null
          ? `${missingShaderCount} missing shader${missingShaderCount === 1 ? '' : 's'}, `
          : '';
      const summary = `Exported missing-shader report (${countDetail}${formatBytes(byteLength)}) to ${filename}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { filename, json, byteLength, missingShaderCount, downloaded, copied },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to export missing shaders');
      return this._finishAction(action, false, `Failed to export missing shaders: ${error}`, {
        error,
      });
    }
  }

  /**
   * Forwards GPU/boot diagnostics telemetry (`bootSummary`, `missingShaders`,
   * `fallbackStats`, `frameStats`) to `gpuBootPanel.update(payload)` when present.
   * Safe no-op returning `null` when `gpuBootPanel` is absent or destroyed.
   *
   * @param {{
   *   bootSummary?: object|null,
   *   missingShaders?: Array|null,
   *   fallbackStats?: object|null,
   *   frameStats?: object|null,
   * }|null|undefined} payload
   * @returns {object|null}
   */
  updateGpuBootTelemetry(payload) {
    if (this._destroyed) {
      return null;
    }
    const panel = this._getGpuBootPanel();
    if (!panel || typeof panel.update !== 'function') {
      return null;
    }
    if (!payload || typeof payload !== 'object') {
      try {
        return typeof panel.getState === 'function' ? panel.getState() : null;
      } catch {
        return null;
      }
    }
    try {
      const state = panel.update(payload);
      return state !== undefined ? state : null;
    } catch {
      return null;
    }
  }

  /**
   * Switches the active input profile via `inputProfiles.setActiveProfile(profileId)`
   * and persists it via `inputProfiles.save()`.
   *
   * @param {string} profileId
   */
  async switchInputProfile(profileId) {
    const action = HUD_ACTIONS.SWITCH_INPUT_PROFILE;
    if (!this._inputProfiles || typeof this._inputProfiles.setActiveProfile !== 'function') {
      return this._finishAction(
        action,
        false,
        'Input profile manager unavailable; cannot switch profile.',
        { error: 'Input profile manager is not configured' },
        'warn'
      );
    }

    try {
      const profile = this._inputProfiles.setActiveProfile(profileId);
      let persisted = false;
      if (typeof this._inputProfiles.save === 'function') {
        persisted = Boolean(this._inputProfiles.save());
      }

      this._syncBoundControlState();

      const label = profile && profile.name ? `${profile.name} (${profile.id})` : String(profileId);
      const summary = `Switched input profile to ${label}${persisted ? ' (saved)' : ''}.`;

      return this._finishAction(
        action,
        true,
        summary,
        { profileId: profile ? profile.id : profileId, profile, persisted },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to switch input profile');
      return this._finishAction(
        action,
        false,
        `Failed to switch input profile: ${error}`,
        { error, profileId }
      );
    }
  }

  /**
   * Toggles the WebGPU G5 smoke demo via `sessionController` (stopping if currently
   * running smoke demo, restarting with `{ preferSmoke: true }` if running Wasm64,
   * or starting with `{ preferSmoke: true }` if idle/stopped/error). Falls back to
   * `smokeDemoRunner.start()` / `stop()` when no `sessionController` is configured.
   */
  async toggleSmokeDemo() {
    const action = HUD_ACTIONS.TOGGLE_SMOKE_DEMO;
    const hasSession =
      this._sessionController &&
      typeof this._sessionController.start === 'function' &&
      typeof this._sessionController.stop === 'function';
    const hasSmokeRunner =
      this._smokeDemoRunner &&
      typeof this._smokeDemoRunner.start === 'function' &&
      typeof this._smokeDemoRunner.stop === 'function';

    if (!hasSession && !hasSmokeRunner) {
      return this._finishAction(
        action,
        false,
        'Session controller unavailable; cannot toggle smoke demo.',
        { error: 'Session controller is not configured' },
        'warn'
      );
    }

    try {
      if (hasSession) {
        const currentState = this._getSessionState();

        if (currentState === SESSION_STATE.RUNNING_SMOKE) {
          const sessionResult = await this._sessionController.stop();
          const state = this._getSessionState();
          const summary = 'Stopped smoke demo session.';
          return this._finishAction(
            action,
            true,
            summary,
            { running: false, state, sessionResult },
            'info'
          );
        }

        const sessionResult =
          currentState === SESSION_STATE.RUNNING_WASM &&
          typeof this._sessionController.restart === 'function'
            ? await this._sessionController.restart({ preferSmoke: true })
            : await this._sessionController.start({ preferSmoke: true });

        const state = this._getSessionState();
        if (sessionResult && sessionResult.ok === false) {
          const error = toErrorMessage(
            sessionResult.error ||
              (Array.isArray(sessionResult.reasons) ? sessionResult.reasons[0] : null),
            'Failed to start smoke demo'
          );
          return this._finishAction(
            action,
            false,
            `Failed to start smoke demo: ${error}`,
            { error, running: false, state, sessionResult }
          );
        }

        const summary = `Started smoke demo session (state: ${state}).`;
        return this._finishAction(
          action,
          true,
          summary,
          { running: state === SESSION_STATE.RUNNING_SMOKE, state, sessionResult },
          'ok'
        );
      }

      // Direct SmokeDemoRunner fallback when no RuntimeSessionController is supplied.
      if (this._smokeDemoRunner.running) {
        this._smokeDemoRunner.stop();
        return this._finishAction(
          action,
          true,
          'Stopped smoke demo runner.',
          { running: false, state: this._getSessionState() },
          'info'
        );
      }

      this._smokeDemoRunner.start();
      return this._finishAction(
        action,
        true,
        'Started smoke demo runner.',
        { running: Boolean(this._smokeDemoRunner.running), state: this._getSessionState() },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to toggle smoke demo');
      return this._finishAction(action, false, `Failed to toggle smoke demo: ${error}`, {
        error,
        state: this._getSessionState(),
      });
    }
  }

  /**
   * Cleanly restarts the active runtime session via `sessionController.restart({ preferSmoke })`.
   *
   * @param {{ preferSmoke?: boolean }} [options]
   */
  async restartSession(options = {}) {
    const action = HUD_ACTIONS.RESTART_SESSION;
    const preferSmoke = Boolean(options && typeof options === 'object' && options.preferSmoke);

    if (!this._sessionController || typeof this._sessionController.restart !== 'function') {
      return this._finishAction(
        action,
        false,
        'Session controller unavailable; cannot restart session.',
        { error: 'Session controller is not configured' },
        'warn'
      );
    }

    try {
      const sessionResult = await this._sessionController.restart({ preferSmoke });
      const state = this._getSessionState();
      const mode = this._getSessionMode();

      if (sessionResult && sessionResult.ok === false) {
        const error = toErrorMessage(
          sessionResult.error ||
            (Array.isArray(sessionResult.reasons) ? sessionResult.reasons[0] : null),
          'Session restart failed'
        );
        return this._finishAction(
          action,
          false,
          `Failed to restart session: ${error}`,
          { error, state, mode, preferSmoke, sessionResult }
        );
      }

      const summary = `Restarted runtime session (mode: ${mode}, state: ${state}).`;
      return this._finishAction(
        action,
        true,
        summary,
        { state, mode, preferSmoke, sessionResult },
        'ok'
      );
    } catch (err) {
      const error = toErrorMessage(err, 'Failed to restart session');
      return this._finishAction(action, false, `Failed to restart session: ${error}`, {
        error,
        state: this._getSessionState(),
        mode: this._getSessionMode(),
        preferSmoke,
      });
    }
  }

  // --- DOM Control Wiring ---

  /**
   * Binds DOM click/change handlers for any provided HUD control elements.
   * Replaces any previously bound controls to avoid duplicate listeners.
   *
   * @param {{
   *   clearCacheBtn?: EventTarget|null,
   *   exportSavesBtn?: EventTarget|null,
   *   recordTraceBtn?: EventTarget|null,
   *   exportTraceBtn?: EventTarget|null,
   *   exportCrashBtn?: EventTarget|null,
   *   exportMissingShadersBtn?: EventTarget|null,
   *   restartBtn?: EventTarget|null,
   *   smokeDemoBtn?: EventTarget|null,
   *   profileSelect?: EventTarget|null,
   * }} [buttonMap]
   * @returns {() => void} Idempotent unbind function.
   */
  bindButtons(buttonMap = {}) {
    if (buttonMap !== undefined && (buttonMap === null || typeof buttonMap !== 'object')) {
      throw new TypeError('bindButtons expects an object mapping control names to DOM elements');
    }

    this.unbindButtons();

    const map = buttonMap || {};
    const handlersByKey = {
      clearCacheBtn: () => {
        void this.clearOpfsCache();
      },
      exportSavesBtn: () => {
        void this.exportSaves();
      },
      recordTraceBtn: () => {
        void this.toggleGpuRecording();
      },
      exportTraceBtn: () => {
        void this.exportGpuTrace();
      },
      exportCrashBtn: () => {
        void this.exportCrashReport();
      },
      exportMissingShadersBtn: () => {
        void this.exportMissingShaders();
      },
      restartBtn: () => {
        void this.restartSession();
      },
      smokeDemoBtn: () => {
        void this.toggleSmokeDemo();
      },
      profileSelect: (event) => {
        const el = map.profileSelect;
        const selected =
          event && event.target && typeof event.target.value === 'string'
            ? event.target.value
            : el && typeof el.value === 'string'
              ? el.value
              : '';
        void this.switchInputProfile(selected);
      },
    };

    for (const [key, eventType] of SUPPORTED_BUTTON_BINDINGS) {
      const element = map[key];
      if (element === undefined || element === null) {
        continue;
      }
      validateEventTarget(element, key);
      const handler = handlersByKey[key];
      element.addEventListener(eventType, handler);
      this._boundListeners.push({ key, element, eventType, handler });
      this._boundElements[key] = element;
    }

    this._syncBoundControlState();

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.unbindButtons();
    };
  }

  /**
   * Removes all DOM event listeners previously attached via `bindButtons()`. Idempotent.
   */
  unbindButtons() {
    for (const entry of this._boundListeners) {
      try {
        entry.element.removeEventListener(entry.eventType, entry.handler);
      } catch {
        // Ignore teardown errors from detached/mock elements.
      }
    }
    this._boundListeners = [];
    this._boundElements = {};
  }

  /**
   * Tears down all bound DOM listeners and marks this controller destroyed. Idempotent.
   */
  destroy() {
    this.unbindButtons();
    this._destroyed = true;
  }

  // --- State Introspection ---

  /**
   * Returns current HUD control states across recording, session lifecycle,
   * active input profile, and last executed action.
   */
  getState() {
    return {
      isRecording: this._isRecording(),
      sessionState: this._getSessionState(),
      sessionMode: this._getSessionMode(),
      activeProfileId: this._getActiveProfileId(),
      lastAction: this._lastAction ? { ...this._lastAction } : null,
      boundControlCount: this._boundListeners.length,
      destroyed: this._destroyed,
    };
  }
}

export default HudController;
