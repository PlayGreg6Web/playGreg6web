import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  INPUT_STATE_BYTES,
  INPUT_OFFSETS,
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  applyDeadzone,
  applyAxialDeadzone,
  applyTriggerDeadzone,
  InputBridge,
} from '../web/input/input_bridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

class MockEventTarget {
  constructor() {
    this._handlers = new Map();
  }

  addEventListener(type, handler) {
    if (!this._handlers.has(type)) {
      this._handlers.set(type, new Set());
    }
    this._handlers.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._handlers.get(type);
    if (!set) return;
    set.delete(handler);
    if (set.size === 0) this._handlers.delete(type);
  }

  dispatch(type, event = {}) {
    const set = this._handlers.get(type);
    if (!set) return;
    for (const fn of [...set]) {
      fn(event);
    }
  }

  listenerCount(type) {
    if (type !== undefined) {
      return this._handlers.get(type)?.size ?? 0;
    }
    let total = 0;
    for (const set of this._handlers.values()) {
      total += set.size;
    }
    return total;
  }
}

function createMockEnv({ gamepads = [], withRaf = false } = {}) {
  const canvas = new MockEventTarget();
  canvas.pointerLockRequests = 0;
  canvas.requestPointerLock = () => {
    canvas.pointerLockRequests += 1;
    return Promise.resolve();
  };

  const doc = new MockEventTarget();
  doc.pointerLockElement = null;
  doc.hidden = false;
  doc.visibilityState = 'visible';

  const win = new MockEventTarget();
  let nextRafId = 1;
  const rafQueue = new Map();
  if (withRaf) {
    win.requestAnimationFrame = (cb) => {
      const id = nextRafId++;
      rafQueue.set(id, cb);
      return id;
    };
    win.cancelAnimationFrame = (id) => {
      rafQueue.delete(id);
    };
  }

  let currentGamepads = gamepads;
  const nav = {
    getGamepads: () => currentGamepads,
  };

  return {
    canvas,
    doc,
    win,
    nav,
    setGamepads(pads) {
      currentGamepads = pads;
    },
    lockPointer() {
      doc.pointerLockElement = canvas;
      doc.dispatch('pointerlockchange');
    },
    unlockPointer() {
      doc.pointerLockElement = null;
      doc.dispatch('pointerlockchange');
    },
    flushRaf() {
      const entries = [...rafQueue.entries()];
      rafQueue.clear();
      for (const [, cb] of entries) {
        cb(16.67);
      }
      return entries.length;
    },
    pendingRafCount() {
      return rafQueue.size;
    },
  };
}

function makeStandardGamepad({
  buttons = {},
  axes = [0, 0, 0, 0],
  connected = true,
  mapping = 'standard',
  index = 0,
} = {}) {
  const btnArray = Array.from({ length: 17 }, (_, i) => {
    const spec = buttons[i];
    if (typeof spec === 'number') {
      return { pressed: spec > 0.5, touched: spec > 0, value: spec };
    }
    if (spec && typeof spec === 'object') {
      return {
        pressed: Boolean(spec.pressed),
        touched: Boolean(spec.touched ?? spec.pressed),
        value: spec.value ?? (spec.pressed ? 1 : 0),
      };
    }
    return { pressed: false, touched: false, value: 0 };
  });

  return {
    id: 'Xbox Wireless Controller (STANDARD GAMEPAD)',
    index,
    connected,
    mapping,
    buttons: btnArray,
    axes: axes.slice(),
  };
}

function approxEqual(actual, expected, eps = 1e-5) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`
  );
}

test('ABI constants and button bitmasks match runtime/abi/playgta6_abi.h and runtime/os/game_input.h byte-for-byte', () => {
  assert.equal(INPUT_STATE_BYTES, 40);
  assert.deepEqual(INPUT_OFFSETS, {
    SEQ: 0,
    BUTTONS: 4,
    MOUSE_DX: 8,
    MOUSE_DY: 12,
    LEFT_STICK_X: 16,
    LEFT_STICK_Y: 20,
    RIGHT_STICK_X: 24,
    RIGHT_STICK_Y: 28,
    LEFT_TRIGGER: 32,
    RIGHT_TRIGGER: 36,
  });

  // Verify C++ headers in repo still agree on sizeof(InputState) == 40 and bitmask shifts.
  const abiHeader = fs.readFileSync(
    path.join(REPO_ROOT, 'runtime/abi/playgta6_abi.h'),
    'utf8'
  );
  assert.match(abiHeader, /static_assert\(sizeof\(InputState\)\s*==\s*40/);

  const gameInputHeader = fs.readFileSync(
    path.join(REPO_ROOT, 'runtime/os/game_input.h'),
    'utf8'
  );
  const expectedCppBits = [
    ['kMoveForward', INPUT_BUTTONS.MOVE_FORWARD, 0],
    ['kMoveBackward', INPUT_BUTTONS.MOVE_BACKWARD, 1],
    ['kMoveLeft', INPUT_BUTTONS.MOVE_LEFT, 2],
    ['kMoveRight', INPUT_BUTTONS.MOVE_RIGHT, 3],
    ['kJump', INPUT_BUTTONS.JUMP, 4],
    ['kSprint', INPUT_BUTTONS.SPRINT, 5],
    ['kGamepadA', INPUT_BUTTONS.A, 8],
    ['kGamepadB', INPUT_BUTTONS.B, 9],
    ['kGamepadX', INPUT_BUTTONS.X, 10],
    ['kGamepadY', INPUT_BUTTONS.Y, 11],
    ['kLeftBumper', INPUT_BUTTONS.LB, 12],
    ['kRightBumper', INPUT_BUTTONS.RB, 13],
    ['kMenu', INPUT_BUTTONS.MENU, 14],
    ['kView', INPUT_BUTTONS.VIEW, 15],
  ];

  for (const [cppName, jsValue, bit] of expectedCppBits) {
    assert.equal(jsValue, 1 << bit, `${cppName} bit mismatch`);
    const re = new RegExp(`constexpr\\s+uint32_t\\s+${cppName}\\s*=\\s*1u\\s*<<\\s*${bit}\\s*;`);
    assert.match(gameInputHeader, re, `Header mismatch for ${cppName}`);
  }

  // Ensure all defined INPUT_BUTTONS bits are distinct powers of two.
  const values = Object.values(INPUT_BUTTONS);
  const seen = new Set();
  for (const v of values) {
    assert.ok(v > 0 && (v & (v - 1)) === 0, `Expected single bit mask, got ${v}`);
    assert.ok(!seen.has(v), `Duplicate button bitmask ${v}`);
    seen.add(v);
  }
});

test('SharedArrayBuffer 40-byte binary layout matches DataView offsets and seqlock protocol', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.ok(bridge.sab instanceof SharedArrayBuffer);
  assert.equal(bridge.sab.byteLength, 40);
  assert.equal(bridge.i32.length, 10);
  assert.equal(bridge.u32.length, 10);
  assert.equal(bridge.f32.length, 10);

  const initial = bridge.getSnapshot();
  assert.equal(initial.seq, 0);
  assert.equal(initial.buttons, 0);
  assert.equal(initial.mouse_dx, 0);
  assert.equal(initial.mouse_dy, 0);
  assert.equal(initial.lx, 0);
  assert.equal(initial.ly, 0);
  assert.equal(initial.rx, 0);
  assert.equal(initial.ry, 0);
  assert.equal(initial.lt, 0);
  assert.equal(initial.rt, 0);

  // Lock pointer and drive every field: mouse delta, buttons, left/right sticks, triggers.
  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 17, movementY: -23 });
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.setGamepads([
    makeStandardGamepad({
      buttons: {
        0: 1, // A
        6: 0.5, // LT
        7: 0.98, // RT
      },
      axes: [0, 0, 0.5, -0.5], // RX = +0.5, RY = +0.5 (Y inverted)
    }),
  ]);
  assert.equal(bridge.pollGamepads(), true);

  // Verify raw little-endian bytes via DataView at exact C++ playgta6::InputState offsets.
  const dv = new DataView(bridge.sab);
  const seq = dv.getUint32(0, true);
  assert.ok(seq > 0 && seq % 2 === 0, `seq should be positive even, got ${seq}`);
  assert.equal(dv.getUint32(4, true), INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.A);
  assert.equal(dv.getInt32(8, true), 17);
  assert.equal(dv.getInt32(12, true), -23);
  approxEqual(dv.getFloat32(16, true), 0.0);
  approxEqual(dv.getFloat32(20, true), 1.0);

  const expectedRight = applyDeadzone(0.5, 0.5);
  approxEqual(dv.getFloat32(24, true), expectedRight.x);
  approxEqual(dv.getFloat32(28, true), expectedRight.y);
  approxEqual(dv.getFloat32(32, true), applyTriggerDeadzone(0.5));
  approxEqual(dv.getFloat32(36, true), 1.0);

  // Seqlock odd sequence (write in progress) causes getSnapshot() to retry and return null.
  Atomics.store(bridge.u32, 0, seq + 1);
  assert.equal(bridge.getSnapshot(4), null);
  Atomics.store(bridge.u32, 0, seq + 2);
  const recovered = bridge.getSnapshot();
  assert.equal(recovered.seq, seq + 2);

  bridge.destroy();
});

test('constructor supports options-only call, custom SAB injection, and Node without DOM globals', () => {
  const customSab = new SharedArrayBuffer(40);
  const bridge = new InputBridge({
    sab: customSab,
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  assert.equal(bridge.sab, customSab);
  assert.equal(bridge.pollGamepads(), false);
  assert.equal(bridge.getSnapshot().seq, 0);
  bridge.destroy();

  assert.throws(
    () => new InputBridge({ sab: new SharedArrayBuffer(16) }),
    RangeError
  );
});

test('keyboard WASD drives normalized left stick axes + digital bits and Xbox key bindings work cleanly', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // W -> forward (+Y)
  env.win.dispatch('keydown', { code: 'KeyW' });
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 1);
  const seqAfterW = snap.seq;

  // Repeat keydown should not bump seq.
  env.win.dispatch('keydown', { code: 'KeyW', repeat: true });
  assert.equal(bridge.getSnapshot().seq, seqAfterW);

  // Unbound key should not bump seq.
  env.win.dispatch('keydown', { code: 'KeyZ' });
  assert.equal(bridge.getSnapshot().seq, seqAfterW);

  // W + D -> diagonal unit vector (1/sqrt(2), 1/sqrt(2))
  env.win.dispatch('keydown', { code: 'KeyD' });
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.MOVE_RIGHT);
  approxEqual(snap.lx, Math.SQRT1_2);
  approxEqual(snap.ly, Math.SQRT1_2);
  approxEqual(Math.hypot(snap.lx, snap.ly), 1.0);

  // Opposing W + S cancels Y axis to 0 while D remains +1
  env.win.dispatch('keydown', { code: 'KeyS' });
  snap = bridge.getSnapshot();
  approxEqual(snap.lx, 1);
  approxEqual(snap.ly, 0);

  // Release W and D -> only S remains (0, -1)
  env.win.dispatch('keyup', { code: 'KeyW' });
  env.win.dispatch('keyup', { code: 'KeyD' });
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_BACKWARD);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, -1);
  env.win.dispatch('keyup', { code: 'KeyS' });

  // Verify all key bindings (Space, Shift, Ctrl, E, F, R, Q, Tab, Escape, Arrows)
  const keysToVerify = [
    ['Space', INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A],
    ['ShiftLeft', INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS],
    ['ControlLeft', INPUT_BUTTONS.B],
    ['KeyR', INPUT_BUTTONS.X],
    ['KeyF', INPUT_BUTTONS.Y],
    ['KeyQ', INPUT_BUTTONS.LB],
    ['KeyE', INPUT_BUTTONS.RB],
    ['Tab', INPUT_BUTTONS.VIEW],
    ['Escape', INPUT_BUTTONS.MENU],
    ['KeyC', INPUT_BUTTONS.RS],
    ['ArrowUp', INPUT_BUTTONS.DPAD_UP],
    ['ArrowDown', INPUT_BUTTONS.DPAD_DOWN],
    ['ArrowLeft', INPUT_BUTTONS.DPAD_LEFT],
    ['ArrowRight', INPUT_BUTTONS.DPAD_RIGHT],
  ];

  for (const [code, expectedMask] of keysToVerify) {
    assert.equal(DEFAULT_KEY_BINDINGS[code], expectedMask);
    env.win.dispatch('keydown', { code });
    assert.equal(bridge.getSnapshot().buttons, expectedMask, `keydown ${code}`);
    env.win.dispatch('keyup', { code });
    assert.equal(bridge.getSnapshot().buttons, 0, `keyup ${code}`);
  }

  // Prevent default on navigation keys while pointer is locked.
  env.lockPointer();
  let prevented = false;
  env.win.dispatch('keydown', {
    code: 'Space',
    preventDefault: () => {
      prevented = true;
    },
  });
  assert.equal(prevented, true);

  bridge.destroy();
});

test('Pointer Lock click, mouse delta accumulation, C++ atomic consume, mouse buttons, and wheel', () => {
  const lockChanges = [];
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    onPointerLockChange: (locked) => lockChanges.push(locked),
  });

  // Canvas click requests pointer lock when unlocked.
  env.canvas.dispatch('click');
  assert.equal(env.canvas.pointerLockRequests, 1);

  // Mousemove & mousedown ignored while unlocked.
  env.win.dispatch('mousemove', { movementX: 50, movementY: -20 });
  env.win.dispatch('mousedown', { button: 0 });
  let snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 0);
  assert.equal(snap.mouse_dy, 0);
  assert.equal(snap.rt, 0);
  assert.equal(snap.seq, 0);

  // Acquire pointer lock.
  env.lockPointer();
  assert.equal(bridge.isPointerLocked(), true);
  assert.deepEqual(lockChanges, [true]);

  // Clicking again while already locked should not re-request pointer lock.
  env.canvas.dispatch('click');
  assert.equal(env.canvas.pointerLockRequests, 1);

  // Accumulate multiple mousemove deltas.
  env.win.dispatch('mousemove', { movementX: 12, movementY: -5 });
  env.win.dispatch('mousemove', { movementX: 8, movementY: -3 });
  snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 20);
  assert.equal(snap.mouse_dy, -8);
  assert.equal(snap.seq, 4); // 2 updates x 2 seqlock bumps

  // Keyboard update MUST NOT clobber unconsumed mouse deltas.
  env.win.dispatch('keydown', { code: 'KeyW' });
  snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 20);
  assert.equal(snap.mouse_dy, -8);

  // Simulate C++ GameInputReader::consume_mouse_delta() atomic exchange(0).
  const consumedDx = Atomics.exchange(bridge.i32, 2, 0);
  const consumedDy = Atomics.exchange(bridge.i32, 3, 0);
  assert.equal(consumedDx, 20);
  assert.equal(consumedDy, -8);
  assert.equal(bridge.getSnapshot().mouse_dx, 0);
  assert.equal(bridge.getSnapshot().mouse_dy, 0);

  // Mouse buttons -> triggers & buttons (0: RT fire, 2: LT aim, 1: RS, 3: LB, 4: RB).
  env.win.dispatch('mousedown', { button: 0 });
  env.win.dispatch('mousedown', { button: 2 });
  env.win.dispatch('mousedown', { button: 1 });
  snap = bridge.getSnapshot();
  approxEqual(snap.rt, 1.0);
  approxEqual(snap.lt, 1.0);
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.RS);

  env.win.dispatch('mouseup', { button: 0 });
  snap = bridge.getSnapshot();
  approxEqual(snap.rt, 0.0);
  approxEqual(snap.lt, 1.0);

  // Scroll wheel tracking while pointer-locked.
  env.win.dispatch('wheel', { deltaY: 120 });
  env.win.dispatch('wheel', { deltaY: -40 });
  assert.equal(bridge.getSnapshot().wheelDelta, 80);
  assert.equal(bridge.consumeWheelDelta(), 80);
  assert.equal(bridge.consumeWheelDelta(), 0);

  // Losing pointer lock releases held mouse buttons/triggers automatically.
  env.unlockPointer();
  assert.equal(bridge.isPointerLocked(), false);
  assert.deepEqual(lockChanges, [true, false]);
  snap = bridge.getSnapshot();
  approxEqual(snap.lt, 0.0);
  approxEqual(snap.rt, 0.0);
  assert.equal(snap.buttons & INPUT_BUTTONS.RS, 0);

  bridge.destroy();
});

test('window blur and document visibilitychange clear stuck keys and mouse buttons', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('keydown', { code: 'Space' });
  env.win.dispatch('mousedown', { button: 0 });

  let snap = bridge.getSnapshot();
  assert.ok(snap.buttons !== 0);
  assert.equal(snap.ly, 1);
  assert.equal(snap.rt, 1);

  // Window loses focus -> all keyboard keys & mouse triggers cleared.
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);

  // Hold keys again and hide document tab -> cleared via visibilitychange.
  env.win.dispatch('keydown', { code: 'KeyD' });
  env.win.dispatch('keydown', { code: 'KeyR' });
  assert.notEqual(bridge.getSnapshot().buttons, 0);

  env.doc.hidden = true;
  env.doc.visibilityState = 'hidden';
  env.doc.dispatch('visibilitychange');

  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);

  bridge.destroy();
});

test('deadzone math matches runtime/os/game_input.cpp normalization and clamps safely', () => {
  // Radial stick deadzone
  assert.deepEqual(applyDeadzone(0.1, -0.08, 0.15, 0.98), { x: 0, y: 0 });

  const diag = applyDeadzone(0.5, 0.5, 0.15, 0.98);
  approxEqual(diag.x, diag.y);
  const expectedMag = (Math.hypot(0.5, 0.5) - 0.15) / (0.98 - 0.15);
  approxEqual(Math.hypot(diag.x, diag.y), expectedMag);

  const clamped = applyDeadzone(2.0, 0.0, 0.15, 0.98);
  approxEqual(clamped.x, 1.0);
  approxEqual(clamped.y, 0.0);

  const clampedDiag = applyDeadzone(1.0, 1.0, 0.15, 0.98);
  approxEqual(Math.hypot(clampedDiag.x, clampedDiag.y), 1.0);
  approxEqual(clampedDiag.x, Math.SQRT1_2);
  approxEqual(clampedDiag.y, Math.SQRT1_2);

  assert.deepEqual(applyDeadzone(Number.NaN, 0.5), { x: 0, y: 0 });
  assert.deepEqual(applyDeadzone(0.5, Number.POSITIVE_INFINITY), { x: 0, y: 0 });

  // Axial deadzone
  assert.equal(applyAxialDeadzone(0.1, 0.15, 0.98), 0);
  approxEqual(applyAxialDeadzone(-0.565, 0.15, 0.98), -0.5);
  assert.equal(applyAxialDeadzone(1.5, 0.15, 0.98), 1);
  assert.equal(applyAxialDeadzone(Number.NaN), 0);

  // Trigger deadzone
  assert.equal(applyTriggerDeadzone(0.03, 0.05, 0.95), 0);
  approxEqual(applyTriggerDeadzone(0.5, 0.05, 0.95), 0.5);
  assert.equal(applyTriggerDeadzone(1.2, 0.05, 0.95), 1);
  assert.equal(applyTriggerDeadzone(Number.NaN), 0);
});

test('Standard Gamepad API maps all 17 buttons, analog triggers, sticks, and merges with keyboard/mouse', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // Verify every W3C Standard Gamepad digital button index (0..5, 8..16).
  const digitalButtons = [
    [0, INPUT_BUTTONS.A],
    [1, INPUT_BUTTONS.B],
    [2, INPUT_BUTTONS.X],
    [3, INPUT_BUTTONS.Y],
    [4, INPUT_BUTTONS.LB],
    [5, INPUT_BUTTONS.RB],
    [8, INPUT_BUTTONS.VIEW],
    [9, INPUT_BUTTONS.MENU],
    [10, INPUT_BUTTONS.LS],
    [11, INPUT_BUTTONS.RS],
    [12, INPUT_BUTTONS.DPAD_UP],
    [13, INPUT_BUTTONS.DPAD_DOWN],
    [14, INPUT_BUTTONS.DPAD_LEFT],
    [15, INPUT_BUTTONS.DPAD_RIGHT],
    [16, INPUT_BUTTONS.GUIDE],
  ];

  for (const [idx, expectedBit] of digitalButtons) {
    env.setGamepads([makeStandardGamepad({ buttons: { [idx]: { pressed: true, value: 1 } } })]);
    assert.equal(bridge.pollGamepads(), true, `button ${idx} press should commit`);
    assert.equal(bridge.getSnapshot().buttons, expectedBit, `button ${idx} bitmask`);
  }

  // Test sticks (with W3C Y-down -> Xbox Y-up inversion) and analog triggers.
  env.setGamepads([
    makeStandardGamepad({
      buttons: {
        0: { pressed: true, value: 1 },
        6: { pressed: true, value: 0.515 },
        7: { pressed: true, value: 0.98 },
      },
      // Left stick: right & up (axes[1] = -0.98 -> ly = +1 after inversion & deadzone)
      // Right stick: inside deadzone -> (0, 0)
      axes: [0.0, -0.98, 0.05, -0.05],
    }),
  ]);

  assert.equal(bridge.pollGamepads(), true);
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 1.0);
  approxEqual(snap.rx, 0);
  approxEqual(snap.ry, 0);
  approxEqual(snap.lt, applyTriggerDeadzone(0.515));
  approxEqual(snap.rt, 1.0);

  // Polling again with identical gamepad state must NOT bump seq.
  const seqSteady = snap.seq;
  assert.equal(bridge.pollGamepads(), false);
  assert.equal(bridge.getSnapshot().seq, seqSteady);

  // Merge check: keyboard buttons OR with gamepad buttons; active keyboard WASD overrides gamepad stick,
  // while zero keyboard axes yield to gamepad axes; triggers take max(mouse, gamepad).
  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyE' }); // RB
  env.win.dispatch('keydown', { code: 'KeyA' }); // MOVE_LEFT -> keyLx = -1, keyLy = 0
  env.win.dispatch('mousedown', { button: 2 });  // Right click -> mouseLt = 1.0

  snap = bridge.getSnapshot();
  assert.equal(
    snap.buttons,
    INPUT_BUTTONS.A | INPUT_BUTTONS.RB | INPUT_BUTTONS.MOVE_LEFT
  );
  approxEqual(snap.lx, -1.0); // Keyboard A overrides zero gamepad X
  approxEqual(snap.ly, 1.0);  // Keyboard Y is 0 -> gamepad Y (+1.0) overrides zero keyboard Y
  approxEqual(snap.lt, 1.0);  // max(mouse 1.0, gamepad 0.5)
  approxEqual(snap.rt, 1.0);  // max(mouse 0.0, gamepad 1.0)

  // Release keyboard & mouse, then disconnect gamepad -> returns cleanly to zero.
  env.win.dispatch('keyup', { code: 'KeyE' });
  env.win.dispatch('keyup', { code: 'KeyA' });
  env.win.dispatch('mouseup', { button: 2 });
  env.setGamepads([null]);
  assert.equal(bridge.pollGamepads(), true);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);

  bridge.destroy();
});

test('requestAnimationFrame polling loop start()/stop() and gamepadconnected auto-start', () => {
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);

  // Explicit start()/stop()
  bridge.start();
  assert.equal(bridge.running, true);
  assert.equal(env.pendingRafCount(), 1);

  env.setGamepads([makeStandardGamepad({ buttons: { 3: 1 } })]); // Y
  env.flushRaf();
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.Y);
  assert.equal(env.pendingRafCount(), 1); // Rescheduled next frame

  bridge.stop();
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);

  // Auto-start on gamepadconnected and auto-stop on gamepaddisconnected
  env.setGamepads([makeStandardGamepad({ buttons: { 2: 1 } })]); // X
  env.win.dispatch('gamepadconnected');
  assert.equal(bridge.running, true);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.X);

  env.setGamepads([null]);
  env.win.dispatch('gamepaddisconnected');
  assert.equal(bridge.running, false);
  assert.equal(bridge.getSnapshot().buttons, 0);

  bridge.destroy();
});

test('destroy() / dispose() cleans up all event listeners, stops rAF loop, and zeros SAB', () => {
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    autoStart: true,
  });

  assert.ok(env.canvas.listenerCount() > 0);
  assert.ok(env.doc.listenerCount() > 0);
  assert.ok(env.win.listenerCount() > 0);
  assert.equal(bridge.running, true);

  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('mousemove', { movementX: 15, movementY: 9 });
  assert.notEqual(bridge.getSnapshot().buttons, 0);

  bridge.destroy();

  assert.equal(bridge.disposed, true);
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);
  assert.equal(env.canvas.listenerCount(), 0);
  assert.equal(env.doc.listenerCount(), 0);
  assert.equal(env.win.listenerCount(), 0);

  const zeroed = bridge.getSnapshot();
  assert.equal(zeroed.buttons, 0);
  assert.equal(zeroed.mouse_dx, 0);
  assert.equal(zeroed.mouse_dy, 0);
  assert.equal(zeroed.lx, 0);
  assert.equal(zeroed.ly, 0);
  const seqAfterDestroy = zeroed.seq;

  // Subsequent events or polls are ignored and dispose() is idempotent.
  env.win.dispatch('keydown', { code: 'Space' });
  env.win.dispatch('mousemove', { movementX: 100, movementY: 100 });
  assert.equal(bridge.pollGamepads(), false);
  bridge.dispose();
  assert.equal(bridge.getSnapshot().seq, seqAfterDestroy);
  assert.equal(bridge.getSnapshot().buttons, 0);
});
