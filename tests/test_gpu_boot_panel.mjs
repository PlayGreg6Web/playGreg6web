import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const panelMod = require('../web/ui/gpu_boot_panel.js');
const fallbackRouterMod = require('../web/workers/shader_fallback_router.js');
import { BootTracer } from '../web/telemetry/boot_tracer.js';

const {
  DEFAULT_MAX_ROWS,
  EXPORT_FORMAT,
  EXPORT_SCHEMA_VERSION,
  BOOT_STAGE_ORDER,
  STAGE_NAMES,
  formatMissingShaderRow,
  buildMissingShaderExport,
  GpuBootPanelController,
} = panelMod;

const { ShaderFallbackRouter, SHADER_STAGE } = fallbackRouterMod;

class MockClassList {
  constructor(el) {
    this._el = el;
  }

  _tokens() {
    return new Set(
      String(this._el.className || '')
        .split(/\s+/)
        .filter(Boolean)
    );
  }

  _write(tokens) {
    this._el.className = Array.from(tokens).join(' ');
  }

  add(...names) {
    const tokens = this._tokens();
    for (const name of names) tokens.add(name);
    this._write(tokens);
  }

  remove(...names) {
    const tokens = this._tokens();
    for (const name of names) tokens.delete(name);
    this._write(tokens);
  }

  contains(name) {
    return this._tokens().has(name);
  }
}

class MockElement {
  constructor(tagName, ownerDocument) {
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.style = {
      display: '',
      width: '',
    };
    this._hidden = false;
    this._textContent = '';
    this.className = '';
    this.classList = new MockClassList(this);
    this._listeners = new Map();
    this.onclick = null;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(value) {
    this._hidden = Boolean(value);
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value ?? '');
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'class') this.className = String(value);
    if (name === 'hidden') this._hidden = true;
  }

  getAttribute(name) {
    if (!this.attributes.has(String(name))) return null;
    return this.attributes.get(String(name));
  }

  removeAttribute(name) {
    this.attributes.delete(String(name));
    if (name === 'hidden') this._hidden = false;
  }

  appendChild(child) {
    if (child.parentNode && child.parentNode !== this) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._listeners.get(type);
    if (set) set.delete(handler);
  }

  listenerCount(type) {
    const set = this._listeners.get(type);
    return set ? set.size : 0;
  }

  dispatch(type, event = {}) {
    const set = this._listeners.get(type);
    if (set) {
      for (const handler of Array.from(set)) handler(event);
    }
    if (type === 'click' && typeof this.onclick === 'function') {
      this.onclick(event);
    }
  }

  querySelector(selector) {
    const matches = this.querySelectorAll(selector);
    return matches.length > 0 ? matches[0] : null;
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        return el.classList.contains(selector.slice(1));
      }
      if (selector.startsWith('style[')) {
        const attr = selector.slice('style['.length, -1);
        return el.tagName === 'STYLE' && el.getAttribute(attr) !== null;
      }
      return el.tagName === selector.toUpperCase();
    };
    const walk = (nodes) => {
      for (const child of nodes) {
        if (match(child)) out.push(child);
        walk(child.children);
      }
    };
    walk(this.children);
    return out;
  }
}

class MockDocument {
  constructor() {
    this.head = new MockElement('head', this);
    this.body = new MockElement('body', this);
    this.documentElement = new MockElement('html', this);
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
  }

  createElement(tag) {
    return new MockElement(tag, this);
  }

  querySelector(selector) {
    return this.head.querySelector(selector) || this.body.querySelector(selector);
  }

  getElementsByTagName(tag) {
    const upper = String(tag).toUpperCase();
    if (upper === 'HEAD') return [this.head];
    if (upper === 'BODY') return [this.body];
    return [];
  }
}

function createPanel(overrides = {}) {
  const documentObj = overrides.documentObj || new MockDocument();
  const container =
    overrides.container !== undefined ? overrides.container : documentObj.body;
  const exportCalls = [];
  const controller = new GpuBootPanelController({
    container,
    documentObj,
    now: () => Date.UTC(2026, 0, 1, 12, 0, 0),
    onExportMissingShaders:
      overrides.onExportMissingShaders !== undefined
        ? overrides.onExportMissingShaders
        : (jsonText, payload) => {
            exportCalls.push({ jsonText, payload });
          },
    ...overrides,
    documentObj,
    container,
  });
  return { controller, documentObj, container, exportCalls };
}

test('exports UMD global, helpers, constants, and GpuBootPanelController', () => {
  assert.equal(DEFAULT_MAX_ROWS, 32);
  assert.equal(EXPORT_FORMAT, 'playgta6-missing-shaders-v1');
  assert.equal(EXPORT_SCHEMA_VERSION, 1);
  assert.ok(Object.isFrozen(BOOT_STAGE_ORDER));
  assert.ok(Object.isFrozen(STAGE_NAMES));
  assert.equal(typeof formatMissingShaderRow, 'function');
  assert.equal(typeof buildMissingShaderExport, 'function');
  assert.equal(typeof GpuBootPanelController, 'function');
  assert.equal(globalThis.PlayGta6GpuBootPanel.GpuBootPanelController, GpuBootPanelController);
  assert.equal(globalThis.PlayGta6GpuBootPanel.formatMissingShaderRow, formatMissingShaderRow);
  assert.equal(globalThis.PlayGta6GpuBootPanel.buildMissingShaderExport, buildMissingShaderExport);
});

test('formatMissingShaderRow normalizes hashes, stages, and hit counts while rejecting invalid entries', () => {
  const r1 = formatMissingShaderRow({
    hash: 0xdeadbeefn,
    stage: 1,
    hitCount: 14,
    firstSeenMs: 12.5,
    lastSeenMs: 90,
  });
  assert.deepEqual(r1, {
    hash: '0x00000000deadbeef',
    stage: 'vertex',
    hits: 14,
    firstSeenMs: 12.5,
    lastSeenMs: 90,
    text: '0x00000000deadbeef | vertex | 14',
  });

  const r2 = formatMissingShaderRow({
    shaderHash: 'CAFEBABE',
    stage: 'PS',
    drawCount: 7,
  });
  assert.equal(r2.hash, '0x00000000cafebabe');
  assert.equal(r2.stage, 'fragment');
  assert.equal(r2.hits, 7);

  const r3 = formatMissingShaderRow({ hash: 255, stage: 'cs' });
  assert.equal(r3.hash, '0x00000000000000ff');
  assert.equal(r3.stage, 'compute');
  assert.equal(r3.hits, 1);

  const r4 = formatMissingShaderRow({ hash: '0x1', stage: 'C:\\Users\\alice\\shader.hlsl', hits: 0 });
  assert.equal(r4.stage, 'unknown');
  assert.equal(r4.hits, 0);

  // Invalid hashes, host paths, and raw byte buffers are rejected cleanly.
  assert.equal(formatMissingShaderRow(null), null);
  assert.equal(formatMissingShaderRow({}), null);
  assert.equal(formatMissingShaderRow({ hash: 'C:\\Users\\alice\\game.rpf', stage: 'vertex' }), null);
  assert.equal(formatMissingShaderRow({ hash: new Uint8Array([1, 2, 3, 4]), stage: 'fragment' }), null);
  assert.equal(formatMissingShaderRow({ hash: -1n, stage: 'compute' }), null);
});

test('buildMissingShaderExport deduplicates, sorts, and strips host paths and raw bytes', () => {
  const payload = buildMissingShaderExport({
    bootSummary: {
      stage: 'C:\\Users\\alice\\GTA6\\game.exe',
      milestonesReached: 4,
      progressPct: 44,
      readyForSplash: false,
      hostPath: '/home/alice/secret/game',
      rawDump: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    },
    frameStats: {
      framesPresented: 3,
      droppedFrames: 1,
      localFile: 'file:///C:/Users/alice/frame.bin',
    },
    fallbackStats: {
      packHits: 15,
      fallbackHits: 3,
      builtinHits: 2,
      rawShaderBuffer: new ArrayBuffer(32),
    },
    missingShaders: [
      {
        hash: '0x10',
        stage: 'vs',
        hitCount: 2,
        sourcePath: 'C:\\Users\\alice\\shaders\\sky.dxil',
        bytecode: new Uint8Array([1, 2, 3, 4]),
      },
      {
        hash: 0x10n,
        stage: 'vertex',
        hits: 5,
        firstSeenMs: 10,
        lastSeenMs: 40,
      },
      {
        hash: '0x20',
        stage: 'fragment',
        hitCount: 12,
        wgslBytes: new Uint8Array([9, 8, 7]),
      },
      {
        hash: '/home/alice/not-a-hash',
        stage: 'compute',
        hits: 99,
      },
    ],
    exportedAt: Date.UTC(2026, 5, 1, 0, 0, 0),
  });

  assert.equal(payload.format, EXPORT_FORMAT);
  assert.equal(payload.schemaVersion, EXPORT_SCHEMA_VERSION);
  assert.equal(payload.exportedAt, '2026-06-01T00:00:00.000Z');
  assert.equal(payload.boot.stage, 'UNKNOWN');
  assert.equal(payload.gpu.framesPresented, 3);
  assert.equal(payload.gpu.droppedFrames, 1);
  assert.equal(payload.gpu.hitRatePct, 75);
  assert.equal(payload.missingShaderCount, 2);
  assert.deepEqual(
    payload.missingShaders.map((r) => [r.hash, r.stage, r.hits]),
    [
      ['0x0000000000000020', 'fragment', 12],
      ['0x0000000000000010', 'vertex', 7],
    ]
  );

  const json = JSON.stringify(payload);
  assert.doesNotMatch(json, /Users|alice|home|file:\/\/|bytecode|rawDump|wgslBytes|sourcePath/i);
});

test('constructor validates options and throws descriptive errors', () => {
  assert.throws(() => new GpuBootPanelController(null), TypeError);
  assert.throws(() => new GpuBootPanelController({ documentObj: {} }), TypeError);
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: 0 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: -4 }),
    RangeError
  );
  assert.throws(
    () => new GpuBootPanelController({ documentObj: new MockDocument(), maxRows: 2.5 }),
    RangeError
  );
  assert.throws(
    () =>
      new GpuBootPanelController({
        documentObj: new MockDocument(),
        onExportMissingShaders: 'not-a-fn',
      }),
    TypeError
  );
});

test('initial render mounts panel into container with accessible defaults and empty table', () => {
  const { controller, documentObj, container } = createPanel();
  const state = controller.getState();

  assert.equal(state.visible, true);
  assert.equal(state.collapsed, false);
  assert.equal(state.destroyed, false);
  assert.equal(state.maxRows, DEFAULT_MAX_ROWS);
  assert.equal(state.root.parentNode, container);
  assert.equal(state.boot.stage, 'NOT_STARTED');
  assert.equal(state.boot.milestonesReached, 0);
  assert.equal(state.boot.progressPct, 0);
  assert.equal(state.boot.readyForSplash, false);

  // Shared style injected once in head.
  assert.equal(documentObj.head.querySelectorAll('style[data-pg6-gpu-boot-panel-style]').length, 1);

  // Default empty-state table row and hidden overflow indicator.
  const root = state.root;
  assert.equal(root.querySelectorAll('.pg6-gpu-panel__empty-row').length, 1);
  assert.equal(root.querySelector('.pg6-gpu-panel__overflow').hidden, true);
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '0.0%');
  assert.equal(root.querySelector('.pg6-gpu-panel__missing-header').textContent, 'Missing Shaders (0)');

  controller.destroy();
});

test('ARIA attributes are set properly on region, toggle button, progressbar, and table', () => {
  const { controller } = createPanel({ ariaLabel: 'Custom GPU Diagnostics' });
  const root = controller.getState().root;
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');
  const exportBtn = root.querySelector('.pg6-gpu-panel__export-btn');
  const body = root.querySelector('.pg6-gpu-panel__body');
  const progressBar = root.querySelector('.pg6-gpu-panel__progress-bar');
  const table = root.querySelector('.pg6-gpu-panel__table');

  assert.equal(root.getAttribute('role'), 'region');
  assert.equal(root.getAttribute('aria-label'), 'Custom GPU Diagnostics');
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(toggleBtn.getAttribute('aria-controls'), body.getAttribute('id'));
  assert.equal(exportBtn.getAttribute('aria-label'), 'Export missing shaders JSON');
  assert.equal(progressBar.getAttribute('role'), 'progressbar');
  assert.equal(progressBar.getAttribute('aria-valuemin'), '0');
  assert.equal(progressBar.getAttribute('aria-valuemax'), '100');
  assert.equal(progressBar.getAttribute('aria-valuenow'), '0');
  assert.equal(table.getAttribute('aria-label'), 'Missing shaders');

  controller.destroy();
});

test('update({ bootSummary }) renders milestone, progressPct, milestonesReached, and readyForSplash', () => {
  const { controller } = createPanel();
  controller.update({
    bootSummary: {
      stage: 'D3D12_CREATE_DEVICE',
      milestonesReached: 7,
      progressPct: 78,
      readyForSplash: false,
    },
  });

  const root = controller.getState().root;
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-stage').textContent, 'D3D12_CREATE_DEVICE');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-milestones').textContent, '7/9');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-progress').textContent, '78%');
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-ready').textContent, 'Pending');
  assert.equal(root.querySelector('.pg6-gpu-panel__progress-bar').getAttribute('aria-valuenow'), '78');
  assert.equal(root.querySelector('.pg6-gpu-panel__progress-fill').style.width, '78%');

  // Derives progressPct, milestonesReached, and readyForSplash when omitted.
  controller.update({
    bootSummary: { stage: 'FIRST_PRESENT' },
  });
  const s2 = controller.getState();
  assert.equal(s2.boot.progressPct, 100);
  assert.equal(s2.boot.milestonesReached, 9);
  assert.equal(s2.boot.readyForSplash, true);
  assert.equal(root.querySelector('.pg6-gpu-panel__boot-ready').textContent, 'Ready');

  controller.destroy();
});

test('update({ frameStats, fallbackStats }) updates Frame-1 GPU counters and hitRatePct formatting', () => {
  const { controller } = createPanel();
  controller.update({
    frameStats: { framesPresented: 42, droppedFrames: 3 },
    fallbackStats: {
      packHits: 2,
      fallbackHits: 1,
      builtinHits: 0,
    },
  });

  const state = controller.getState();
  const root = state.root;
  assert.equal(state.frameStats.framesPresented, 42);
  assert.equal(state.frameStats.droppedFrames, 3);
  assert.equal(state.fallbackStats.packHits, 2);
  assert.equal(state.fallbackStats.fallbackHits, 1);
  assert.equal(state.fallbackStats.builtinHits, 0);
  assert.equal(state.fallbackStats.hitRatePct, 66.67);
  assert.equal(state.hitRateText, '66.7%');

  assert.equal(root.querySelector('.pg6-gpu-panel__frames-presented').textContent, '42');
  assert.equal(root.querySelector('.pg6-gpu-panel__dropped-frames').textContent, '3');
  assert.equal(root.querySelector('.pg6-gpu-panel__pack-hits').textContent, '2');
  assert.equal(root.querySelector('.pg6-gpu-panel__fallback-hits').textContent, '1');
  assert.equal(root.querySelector('.pg6-gpu-panel__builtin-hits').textContent, '0');
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '66.7%');

  // Explicit hitRatePct overrides derived computation and clamps to [0, 100].
  controller.update({
    fallbackStats: { packHits: 10, fallbackHits: 0, builtinHits: 0, hitRatePct: 120 },
  });
  assert.equal(controller.getState().fallbackStats.hitRatePct, 100);
  assert.equal(root.querySelector('.pg6-gpu-panel__hit-rate').textContent, '100.0%');

  controller.destroy();
});

test('partial update() preserves previously populated telemetry sections', () => {
  const { controller } = createPanel();
  controller.update({
    bootSummary: { stage: 'GDK_INIT', milestonesReached: 6, progressPct: 67, readyForSplash: false },
    frameStats: { framesPresented: 1, droppedFrames: 0 },
  });

  controller.update({
    missingShaders: [{ hash: '0xabc', stage: 'vertex', hits: 4 }],
  });

  const state = controller.getState();
  assert.equal(state.boot.stage, 'GDK_INIT');
  assert.equal(state.frameStats.framesPresented, 1);
  assert.equal(state.totalMissingShaders, 1);
  assert.equal(state.missingShaders[0].hash, '0x0000000000000abc');

  controller.destroy();
});

test('missing-shader table deduplicates entries and sorts by hits descending then hash ascending', () => {
  const { controller } = createPanel();
  controller.update({
    missingShaders: [
      { hash: '0x0000000000000002', stage: 'fragment', hitCount: 3 },
      { hash: '0x1', stage: 'vs', hits: 5 },
      { hash: 1n, stage: 'vertex', hitCount: 4 }, // merges with 0x1 vertex -> 9
      { hash: '0x0000000000000003', stage: 'compute', hits: 9 }, // ties with 0x1 -> sorted after 0x1
      { hash: '0x1', stage: 'fragment', hits: 1 }, // distinct stage -> stays separate
    ],
  });

  const state = controller.getState();
  assert.equal(state.totalMissingShaders, 4);
  assert.deepEqual(
    state.missingShaders.map((r) => [r.hash, r.stage, r.hits]),
    [
      ['0x0000000000000001', 'vertex', 9],
      ['0x0000000000000003', 'compute', 9],
      ['0x0000000000000002', 'fragment', 3],
      ['0x0000000000000001', 'fragment', 1],
    ]
  );

  const rows = state.root.querySelectorAll('.pg6-gpu-panel__shader-row');
  assert.equal(rows.length, 4);
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-hash').textContent, '0x0000000000000001');
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-stage').textContent, 'vertex');
  assert.equal(rows[0].querySelector('.pg6-gpu-panel__cell-hits').textContent, '9');
  assert.equal(rows[3].querySelector('.pg6-gpu-panel__cell-hits').textContent, '1');

  controller.destroy();
});

test('missing-shader table caps rendered rows to maxRows and shows +N more overflow indicator', () => {
  const { controller } = createPanel({ maxRows: 3 });
  const entries = [];
  for (let i = 1; i <= 7; i += 1) {
    entries.push({ hash: BigInt(i), stage: 'fragment', hits: i * 10 });
  }
  controller.update({ missingShaders: entries });

  const state = controller.getState();
  const root = state.root;
  assert.equal(state.totalMissingShaders, 7);
  assert.equal(state.visibleMissingShaders.length, 3);
  assert.equal(state.overflowCount, 4);
  assert.equal(root.querySelectorAll('.pg6-gpu-panel__shader-row').length, 3);
  assert.equal(
    root.querySelectorAll('.pg6-gpu-panel__shader-row')[0].querySelector('.pg6-gpu-panel__cell-hits')
      .textContent,
    '70'
  );

  const overflowEl = root.querySelector('.pg6-gpu-panel__overflow');
  assert.equal(overflowEl.hidden, false);
  assert.equal(overflowEl.textContent, '+4 more');

  // Shrinking back within maxRows hides the overflow indicator.
  controller.update({ missingShaders: entries.slice(0, 2) });
  assert.equal(controller.getState().overflowCount, 0);
  assert.equal(overflowEl.hidden, true);
  assert.equal(overflowEl.textContent, '');

  controller.destroy();
});

test('export button and exportMissingShadersJson() invoke callback with full sanitized JSON', () => {
  const { controller, exportCalls } = createPanel({ maxRows: 2 });
  controller.update({
    bootSummary: { stage: 'BINK_OPEN', milestonesReached: 8, progressPct: 89, readyForSplash: false },
    frameStats: { framesPresented: 5, droppedFrames: 0 },
    fallbackStats: { packHits: 8, fallbackHits: 2, builtinHits: 0 },
    missingShaders: [
      { hash: '0xaa', stage: 'vertex', hits: 10, hostPath: 'C:\\Users\\secret\\a.bin' },
      { hash: '0xbb', stage: 'fragment', hits: 6, rawBytes: new Uint8Array([1, 2, 3]) },
      { hash: '0xcc', stage: 'compute', hits: 2 },
    ],
  });

  const exportBtn = controller.getState().root.querySelector('.pg6-gpu-panel__export-btn');
  exportBtn.dispatch('click');

  assert.equal(exportCalls.length, 1);
  const { jsonText, payload } = exportCalls[0];
  assert.deepEqual(JSON.parse(jsonText), payload);
  // Export contains all 3 deduplicated shaders even though table is capped at maxRows=2.
  assert.equal(payload.missingShaderCount, 3);
  assert.equal(payload.missingShaders.length, 3);
  assert.equal(payload.boot.stage, 'BINK_OPEN');
  assert.equal(payload.gpu.hitRatePct, 80);
  assert.doesNotMatch(jsonText, /Users|secret|hostPath|rawBytes/i);

  // Direct API call returns JSON string and invokes callback.
  const directJson = controller.exportMissingShadersJson();
  assert.equal(exportCalls.length, 2);
  assert.equal(JSON.parse(directJson).missingShaderCount, 3);
  assert.equal(controller.getState().exportCount, 2);

  controller.destroy();
});

test('toggle(), header toggle button, show(), and hide() update visibility and collapsed ARIA state', () => {
  const { controller } = createPanel();
  const root = controller.getState().root;
  const body = root.querySelector('.pg6-gpu-panel__body');
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');

  assert.equal(controller.getState().collapsed, false);
  assert.equal(body.hidden, false);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(root.getAttribute('data-collapsed'), 'false');

  // Toggle via method collapses body.
  controller.toggle();
  assert.equal(controller.getState().collapsed, true);
  assert.equal(body.hidden, true);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'false');
  assert.equal(root.getAttribute('data-collapsed'), 'true');
  assert.equal(toggleBtn.textContent, 'Expand');

  // Toggle via button click expands body.
  toggleBtn.dispatch('click');
  assert.equal(controller.getState().collapsed, false);
  assert.equal(body.hidden, false);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(toggleBtn.textContent, 'Collapse');

  // hide() / show() control panel visibility independently of collapsed state.
  controller.toggle(true);
  controller.hide();
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().collapsed, true);
  assert.equal(root.hidden, true);
  assert.equal(root.style.display, 'none');

  controller.show();
  assert.equal(controller.getState().visible, true);
  assert.equal(controller.getState().collapsed, true);
  assert.equal(root.hidden, false);
  assert.equal(root.style.display, '');

  controller.destroy();
});

test('error isolation prevents callback failures and malformed snapshots from throwing', () => {
  const { controller } = createPanel({
    onExportMissingShaders() {
      throw new Error('export callback exploded');
    },
  });

  // Throwing export callback must not throw out of exportMissingShadersJson() or click.
  let returnedJson = null;
  assert.doesNotThrow(() => {
    returnedJson = controller.exportMissingShadersJson();
  });
  assert.equal(JSON.parse(returnedJson).format, EXPORT_FORMAT);

  const exportBtn = controller.getState().root.querySelector('.pg6-gpu-panel__export-btn');
  assert.doesNotThrow(() => exportBtn.dispatch('click'));

  // Malformed snapshots and throwing property getters must never crash update().
  assert.doesNotThrow(() => controller.update(null));
  assert.doesNotThrow(() => controller.update(undefined));
  assert.doesNotThrow(() => controller.update('bad-input'));
  assert.doesNotThrow(() =>
    controller.update({
      get bootSummary() {
        throw new Error('getter fault');
      },
      missingShaders: [
        {
          get hash() {
            throw new Error('row getter fault');
          },
        },
        { hash: '0x42', stage: 'compute', hits: 3 },
      ],
    })
  );
  assert.equal(controller.getState().totalMissingShaders, 1);
  assert.equal(controller.getState().missingShaders[0].hash, '0x0000000000000042');

  controller.destroy();
});

test('destroy() is idempotent, removes DOM listeners and node, and guards post-destroy calls', () => {
  const { controller, container, exportCalls } = createPanel();
  const root = controller.getState().root;
  const toggleBtn = root.querySelector('.pg6-gpu-panel__toggle-btn');
  const exportBtn = root.querySelector('.pg6-gpu-panel__export-btn');

  assert.equal(root.parentNode, container);
  assert.equal(toggleBtn.listenerCount('click'), 1);
  assert.equal(exportBtn.listenerCount('click'), 1);

  controller.destroy();
  assert.equal(controller.getState().destroyed, true);
  assert.equal(controller.getState().visible, false);
  assert.equal(controller.getState().root, null);
  assert.equal(root.parentNode, null);
  assert.equal(toggleBtn.listenerCount('click'), 0);
  assert.equal(exportBtn.listenerCount('click'), 0);

  // Clicking detached buttons after destroy does nothing.
  toggleBtn.dispatch('click');
  exportBtn.dispatch('click');
  assert.equal(exportCalls.length, 0);

  // Idempotent destroy and safe post-destroy methods.
  assert.doesNotThrow(() => controller.destroy());
  assert.doesNotThrow(() => controller.update({ bootSummary: { stage: 'PGIM_LOAD' } }));
  assert.doesNotThrow(() => controller.toggle());
  assert.doesNotThrow(() => controller.show());
  assert.doesNotThrow(() => controller.hide());
  assert.doesNotThrow(() => controller.exportMissingShadersJson());
  assert.equal(exportCalls.length, 0);
});

test('integrates directly with live BootTracer and ShaderFallbackRouter snapshots', () => {
  let clockMs = 100;
  const tracer = new BootTracer({ now: () => clockMs, startMs: 100 });
  tracer.markMilestone('PGIM_LOAD');
  clockMs += 50;
  tracer.markMilestone('D3D12_CREATE_DEVICE');

  const pack = new Map([
    [
      '0x0000000000000001',
      '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }',
    ],
  ]);
  const router = new ShaderFallbackRouter({ packLoader: pack, now: () => clockMs });
  router.resolve(1n, SHADER_STAGE.VERTEX); // pack hit
  router.resolve(0xdeadn, SHADER_STAGE.FRAGMENT); // miss 1
  router.resolve(0xdeadn, SHADER_STAGE.FRAGMENT); // miss 2
  router.resolve(0xbeefn, SHADER_STAGE.COMPUTE); // miss 1

  const { controller, exportCalls } = createPanel();
  controller.update({
    bootSummary: tracer.getSummary(),
    missingShaders: router.getMissingReport(),
    fallbackStats: router.getStats(),
    frameStats: { framesPresented: 1, droppedFrames: 0 },
  });

  const state = controller.getState();
  assert.equal(state.boot.stage, 'D3D12_CREATE_DEVICE');
  assert.equal(state.boot.milestonesReached, 2);
  assert.equal(state.boot.progressPct, Math.round((7 / 9) * 100));
  assert.equal(state.fallbackStats.packHits, 1);
  assert.equal(state.fallbackStats.builtinHits, 3);
  assert.equal(state.fallbackStats.hitRatePct, 25);
  assert.equal(state.totalMissingShaders, 2);
  assert.equal(state.missingShaders[0].hash, '0x000000000000dead');
  assert.equal(state.missingShaders[0].hits, 2);
  assert.equal(state.missingShaders[1].hash, '0x000000000000beef');
  assert.equal(state.missingShaders[1].hits, 1);

  const json = controller.exportMissingShadersJson();
  assert.equal(exportCalls.length, 1);
  const parsed = JSON.parse(json);
  assert.equal(parsed.missingShaderCount, 2);
  assert.equal(parsed.gpu.hitRatePct, 25);

  controller.destroy();
});
