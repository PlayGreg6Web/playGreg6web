// Writes playgta6::InputState (40 bytes, runtime/abi/playgta6_abi.h & docs/wasm_abi.md)
// into a SharedArrayBuffer using atomic writes + even/odd seqlock framing compatible with
// both SeqlockMode::MonotonicSeq and SeqlockMode::EvenOddSeqlock in runtime/os/game_input.cpp.

export const INPUT_STATE_BYTES = 40;

export const INPUT_OFFSETS = Object.freeze({
  SEQ: 0,            // uint32_t (u32[0])
  BUTTONS: 4,        // uint32_t (u32[1])
  MOUSE_DX: 8,       // int32_t  (i32[2])
  MOUSE_DY: 12,      // int32_t  (i32[3])
  LEFT_STICK_X: 16,  // float32  (f32[4])
  LEFT_STICK_Y: 20,  // float32  (f32[5])
  RIGHT_STICK_X: 24, // float32  (f32[6])
  RIGHT_STICK_Y: 28, // float32  (f32[7])
  LEFT_TRIGGER: 32,  // float32  (f32[8])
  RIGHT_TRIGGER: 36, // float32  (f32[9])
});

// Button bitmasks matching runtime/os/game_input.h (bits 0..5 & 8..15) plus
// non-overlapping Xbox GameInput extensions (bits 16..22).
export const INPUT_BUTTONS = Object.freeze({
  // Movement & legacy keyboard bits 0..5 (runtime/os/game_input.h)
  MOVE_FORWARD: 1 << 0,   // kMoveForward  (KeyW)
  MOVE_BACKWARD: 1 << 1,  // kMoveBackward (KeyS)
  MOVE_LEFT: 1 << 2,      // kMoveLeft     (KeyA)
  MOVE_RIGHT: 1 << 3,     // kMoveRight    (KeyD)
  JUMP: 1 << 4,           // kJump         (Space)
  SPRINT: 1 << 5,         // kSprint       (ShiftLeft)

  // Xbox GameInput buttons bits 8..15 (runtime/os/game_input.h)
  A: 1 << 8,              // kGamepadA
  B: 1 << 9,              // kGamepadB
  X: 1 << 10,             // kGamepadX
  Y: 1 << 11,             // kGamepadY
  LB: 1 << 12,            // kLeftBumper
  RB: 1 << 13,            // kRightBumper
  MENU: 1 << 14,          // kMenu (Start)
  VIEW: 1 << 15,          // kView (Back / Select)

  // Extended Xbox GameInput buttons bits 16..22
  LS: 1 << 16,            // Left Stick press (L3)
  RS: 1 << 17,            // Right Stick press (R3)
  DPAD_UP: 1 << 18,       // D-Pad Up
  DPAD_DOWN: 1 << 19,     // D-Pad Down
  DPAD_LEFT: 1 << 20,     // D-Pad Left
  DPAD_RIGHT: 1 << 21,    // D-Pad Right
  GUIDE: 1 << 22,         // Xbox Guide / Home button
});

export const DEFAULT_KEY_BINDINGS = Object.freeze({
  KeyW: INPUT_BUTTONS.MOVE_FORWARD,
  KeyS: INPUT_BUTTONS.MOVE_BACKWARD,
  KeyA: INPUT_BUTTONS.MOVE_LEFT,
  KeyD: INPUT_BUTTONS.MOVE_RIGHT,
  Space: INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A,
  ShiftLeft: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
  ShiftRight: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
  ControlLeft: INPUT_BUTTONS.B,
  ControlRight: INPUT_BUTTONS.B,
  KeyC: INPUT_BUTTONS.RS,
  KeyR: INPUT_BUTTONS.X,
  KeyF: INPUT_BUTTONS.Y,
  KeyQ: INPUT_BUTTONS.LB,
  KeyE: INPUT_BUTTONS.RB,
  Tab: INPUT_BUTTONS.VIEW,
  Escape: INPUT_BUTTONS.MENU,
  ArrowUp: INPUT_BUTTONS.DPAD_UP,
  ArrowDown: INPUT_BUTTONS.DPAD_DOWN,
  ArrowLeft: INPUT_BUTTONS.DPAD_LEFT,
  ArrowRight: INPUT_BUTTONS.DPAD_RIGHT,
});

// W3C Standard Gamepad button index -> INPUT_BUTTONS bitmask (indices 6 & 7 are analog triggers).
const STANDARD_GAMEPAD_BUTTON_MAP = Object.freeze([
  INPUT_BUTTONS.A,          // 0: Bottom face button (A)
  INPUT_BUTTONS.B,          // 1: Right face button (B)
  INPUT_BUTTONS.X,          // 2: Left face button (X)
  INPUT_BUTTONS.Y,          // 3: Top face button (Y)
  INPUT_BUTTONS.LB,         // 4: Left bumper (LB)
  INPUT_BUTTONS.RB,         // 5: Right bumper (RB)
  0,                        // 6: Left trigger (analog -> lt)
  0,                        // 7: Right trigger (analog -> rt)
  INPUT_BUTTONS.VIEW,       // 8: Back / View
  INPUT_BUTTONS.MENU,       // 9: Start / Menu
  INPUT_BUTTONS.LS,         // 10: Left stick press (L3)
  INPUT_BUTTONS.RS,         // 11: Right stick press (R3)
  INPUT_BUTTONS.DPAD_UP,    // 12: D-Pad Up
  INPUT_BUTTONS.DPAD_DOWN,  // 13: D-Pad Down
  INPUT_BUTTONS.DPAD_LEFT,  // 14: D-Pad Left
  INPUT_BUTTONS.DPAD_RIGHT, // 15: D-Pad Right
  INPUT_BUTTONS.GUIDE,      // 16: Home / Guide
]);

const PREVENT_DEFAULT_KEYS = new Set([
  'Space',
  'Tab',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
]);

const _f32ScratchBuf = new ArrayBuffer(4);
const _f32Scratch = new Float32Array(_f32ScratchBuf);
const _i32Scratch = new Int32Array(_f32ScratchBuf);

function f32ToI32(value) {
  _f32Scratch[0] = value;
  return _i32Scratch[0];
}

function i32ToF32(bits) {
  _i32Scratch[0] = bits;
  return _f32Scratch[0];
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return value < min ? min : value > max ? max : value;
}

/**
 * Radial 2D stick deadzone normalization matching playgta6::os::normalize_stick.
 * Maps stick magnitude in [innerDeadzone, outerDeadzone] -> [0, 1] while preserving direction.
 */
export function applyDeadzone(x, y, innerDeadzone = 0.15, outerDeadzone = 0.98) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return { x: 0, y: 0 };
  }
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);

  const mag = Math.hypot(x, y);
  if (mag <= inner || mag <= 0) {
    return { x: 0, y: 0 };
  }

  const clampedMag = Math.min(mag, outer);
  const normalizedMag = clamp((clampedMag - inner) / (outer - inner), 0, 1);
  const scale = normalizedMag / mag;
  return {
    x: clamp(x * scale, -1, 1),
    y: clamp(y * scale, -1, 1),
  };
}

/**
 * Axial 1D deadzone normalization with rescaling into [-1, 1].
 */
export function applyAxialDeadzone(value, innerDeadzone = 0.15, outerDeadzone = 0.98) {
  if (!Number.isFinite(value)) return 0;
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);
  const abs = Math.abs(value);
  if (abs <= inner) return 0;
  const normalized = clamp((Math.min(abs, outer) - inner) / (outer - inner), 0, 1);
  return value < 0 ? -normalized : normalized;
}

/**
 * Linear trigger deadzone normalization matching playgta6::os::normalize_trigger.
 * Maps trigger value in [innerDeadzone, outerDeadzone] -> [0, 1].
 */
export function applyTriggerDeadzone(value, innerDeadzone = 0.05, outerDeadzone = 0.98) {
  if (!Number.isFinite(value)) return 0;
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);
  if (value <= inner) return 0;
  if (value >= outer) return 1;
  return clamp((value - inner) / (outer - inner), 0, 1);
}

function isGamepadButtonPressed(button) {
  if (button == null) return false;
  if (typeof button === 'number') return button > 0.5;
  return Boolean(button.pressed || (typeof button.value === 'number' && button.value > 0.5));
}

function readGamepadTrigger(button) {
  if (button == null) return 0;
  if (typeof button === 'number') return clamp(button, 0, 1);
  if (typeof button.value === 'number' && Number.isFinite(button.value)) {
    return clamp(button.value, 0, 1);
  }
  return button.pressed ? 1 : 0;
}

export class InputBridge {
  constructor(canvasOrOptions = null, options = {}) {
    let canvas = canvasOrOptions;
    let opts = options;

    // Support both `new InputBridge(canvas, options)` and `new InputBridge({ canvas, ... })`.
    if (
      canvasOrOptions !== null &&
      typeof canvasOrOptions === 'object' &&
      typeof canvasOrOptions.addEventListener !== 'function' &&
      arguments.length === 1
    ) {
      opts = canvasOrOptions;
      canvas = opts.canvas ?? null;
    }

    opts = opts || {};

    this._canvas = canvas || null;
    this._window =
      opts.windowObj !== undefined
        ? opts.windowObj
        : typeof window !== 'undefined'
          ? window
          : null;
    this._document =
      opts.documentObj !== undefined
        ? opts.documentObj
        : typeof document !== 'undefined'
          ? document
          : null;
    this._navigator =
      opts.navigatorObj !== undefined
        ? opts.navigatorObj
        : typeof navigator !== 'undefined'
          ? navigator
          : null;

    if (opts.sab) {
      if (!(opts.sab instanceof SharedArrayBuffer) || opts.sab.byteLength < INPUT_STATE_BYTES) {
        throw new RangeError(`sab must be a SharedArrayBuffer of at least ${INPUT_STATE_BYTES} bytes`);
      }
      this.sab = opts.sab;
    } else {
      this.sab = new SharedArrayBuffer(INPUT_STATE_BYTES);
    }

    this.i32 = new Int32Array(this.sab);
    this.u32 = new Uint32Array(this.sab);
    this.f32 = new Float32Array(this.sab);
    this.view = new DataView(this.sab);

    this._keyBindings = opts.keyBindings
      ? { ...DEFAULT_KEY_BINDINGS, ...opts.keyBindings }
      : { ...DEFAULT_KEY_BINDINGS };

    this._stickDeadzone = opts.stickDeadzone ?? 0.15;
    this._stickOuterDeadzone = opts.stickOuterDeadzone ?? 0.98;
    this._triggerDeadzone = opts.triggerDeadzone ?? 0.05;
    this._triggerOuterDeadzone = opts.triggerOuterDeadzone ?? 0.98;
    this._invertGamepadY = opts.invertGamepadY ?? true;
    this._gamepadIndex = opts.gamepadIndex ?? null;
    this._allowUnlockedMouseButtons = Boolean(opts.allowUnlockedMouseButtons);
    this._autoStartOnGamepad = opts.autoStart !== false;
    this._onPointerLockChangeCallback =
      typeof opts.onPointerLockChange === 'function' ? opts.onPointerLockChange : null;

    // Input source state
    this._pressedKeys = new Set();
    this._keyButtons = 0;
    this._keyLx = 0;
    this._keyLy = 0;

    this._mouseButtonsMask = 0; // bitmask of raw DOM mouse buttons currently held
    this._mouseButtonBits = 0;  // mapped INPUT_BUTTONS bits from mouse
    this._mouseLt = 0;
    this._mouseRt = 0;
    this._wheelDelta = 0;

    this._gpButtons = 0;
    this._gpLx = 0;
    this._gpLy = 0;
    this._gpRx = 0;
    this._gpRy = 0;
    this._gpLt = 0;
    this._gpRt = 0;

    // Last committed merged state (for change detection in pollGamepads)
    this.buttons = 0;
    this._lastLx = 0;
    this._lastLy = 0;
    this._lastRx = 0;
    this._lastRy = 0;
    this._lastLt = 0;
    this._lastRt = 0;

    this.pointerLocked = Boolean(
      this._canvas && this._document && this._document.pointerLockElement === this._canvas
    );
    this._rafId = null;
    this._manualStart = false;
    this.running = false;
    this.disposed = false;

    this._listeners = [];
    this._bindEvents();

    if (opts.autoStart === true) {
      this.start();
    }
  }

  _addListener(target, event, handler, options) {
    if (!target || typeof target.addEventListener !== 'function') return;
    target.addEventListener(event, handler, options);
    this._listeners.push({ target, event, handler, options });
  }

  _bindEvents() {
    this._addListener(this._canvas, 'click', () => {
      if (this.disposed || !this._canvas) return;
      if (!this.isPointerLocked() && typeof this._canvas.requestPointerLock === 'function') {
        try {
          const result = this._canvas.requestPointerLock();
          if (result && typeof result.catch === 'function') {
            result.catch(() => {});
          }
        } catch {
          // Pointer lock may reject when invoked outside a user gesture.
        }
      }
    });

    this._addListener(this._canvas, 'contextmenu', (e) => {
      if (this.isPointerLocked() && typeof e?.preventDefault === 'function') {
        e.preventDefault();
      }
    });

    this._addListener(this._document, 'pointerlockchange', () => {
      const nowLocked = Boolean(
        this._canvas && this._document && this._document.pointerLockElement === this._canvas
      );
      const changed = nowLocked !== this.pointerLocked;
      this.pointerLocked = nowLocked;
      if (!nowLocked && this._mouseButtonsMask !== 0) {
        this._clearMouseButtons();
        this._commitMergedState(true);
      }
      if (changed && this._onPointerLockChangeCallback) {
        this._onPointerLockChangeCallback(nowLocked);
      }
    });

    this._addListener(this._window, 'mousemove', (e) => {
      if (this.disposed || !this.isPointerLocked()) return;
      const dx = (e?.movementX ?? 0) | 0;
      const dy = (e?.movementY ?? 0) | 0;
      if (dx === 0 && dy === 0) return;

      // Even/odd seqlock framing around atomic mouse delta accumulation.
      // Matches Atomics.exchange(0) in runtime/os/game_input.cpp::consume_mouse_delta().
      Atomics.add(this.u32, 0, 1);
      if (dx !== 0) Atomics.add(this.i32, 2, dx);
      if (dy !== 0) Atomics.add(this.i32, 3, dy);
      Atomics.add(this.u32, 0, 1);
    });

    this._addListener(this._window, 'mousedown', (e) => {
      if (this.disposed) return;
      if (!this.isPointerLocked() && !this._allowUnlockedMouseButtons) return;
      const btn = e?.button;
      if (typeof btn !== 'number' || btn < 0 || btn > 4) return;
      const bit = 1 << btn;
      if ((this._mouseButtonsMask & bit) !== 0) return;
      this._mouseButtonsMask |= bit;
      this._recomputeMouseButtons();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'mouseup', (e) => {
      if (this.disposed) return;
      const btn = e?.button;
      if (typeof btn !== 'number' || btn < 0 || btn > 4) return;
      const bit = 1 << btn;
      if ((this._mouseButtonsMask & bit) === 0) return;
      this._mouseButtonsMask &= ~bit;
      this._recomputeMouseButtons();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'wheel', (e) => {
      if (this.disposed || !this.isPointerLocked()) return;
      const deltaY = Number.isFinite(e?.deltaY) ? e.deltaY : 0;
      if (deltaY !== 0) {
        this._wheelDelta += deltaY;
      }
    });

    this._addListener(this._window, 'keydown', (e) => {
      if (this.disposed || !e || !e.code) return;
      const code = e.code;
      const isWasd = code === 'KeyW' || code === 'KeyS' || code === 'KeyA' || code === 'KeyD';
      const binding = this._keyBindings[code] || 0;
      if (!binding && !isWasd) return;

      if (
        this.isPointerLocked() &&
        PREVENT_DEFAULT_KEYS.has(code) &&
        typeof e.preventDefault === 'function'
      ) {
        e.preventDefault();
      }

      if (e.repeat && this._pressedKeys.has(code)) return;
      if (this._pressedKeys.has(code)) return;

      this._pressedKeys.add(code);
      this._recomputeKeyboardState();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'keyup', (e) => {
      if (this.disposed || !e || !e.code) return;
      if (!this._pressedKeys.delete(e.code)) return;
      this._recomputeKeyboardState();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'blur', () => {
      this._clearTransientInputs();
    });

    this._addListener(this._document, 'visibilitychange', () => {
      const hidden =
        Boolean(this._document?.hidden) || this._document?.visibilityState === 'hidden';
      if (hidden) {
        this._clearTransientInputs();
      }
    });

    this._addListener(this._window, 'gamepadconnected', () => {
      if (this.disposed) return;
      this.pollGamepads();
      if (this._autoStartOnGamepad && !this.running) {
        this._startLoop(false);
      }
    });

    this._addListener(this._window, 'gamepaddisconnected', () => {
      if (this.disposed) return;
      this.pollGamepads();
      if (!this._manualStart && !this._selectGamepad()) {
        this.stop();
      }
    });
  }

  isPointerLocked() {
    if (this._canvas && this._document && 'pointerLockElement' in this._document) {
      this.pointerLocked = this._document.pointerLockElement === this._canvas;
    }
    return this.pointerLocked;
  }

  _recomputeKeyboardState() {
    let mask = 0;
    for (const code of this._pressedKeys) {
      mask |= this._keyBindings[code] || 0;
    }
    this._keyButtons = mask >>> 0;

    let kx =
      (this._pressedKeys.has('KeyD') ? 1 : 0) - (this._pressedKeys.has('KeyA') ? 1 : 0);
    let ky =
      (this._pressedKeys.has('KeyW') ? 1 : 0) - (this._pressedKeys.has('KeyS') ? 1 : 0);

    if (kx !== 0 && ky !== 0) {
      kx *= Math.SQRT1_2;
      ky *= Math.SQRT1_2;
    }
    this._keyLx = kx;
    this._keyLy = ky;
  }

  _recomputeMouseButtons() {
    const m = this._mouseButtonsMask;
    this._mouseRt = (m & (1 << 0)) !== 0 ? 1 : 0; // Left click -> Right Trigger (Fire)
    this._mouseLt = (m & (1 << 2)) !== 0 ? 1 : 0; // Right click -> Left Trigger (Aim)

    let bits = 0;
    if ((m & (1 << 1)) !== 0) bits |= INPUT_BUTTONS.RS; // Middle click -> RS
    if ((m & (1 << 3)) !== 0) bits |= INPUT_BUTTONS.LB; // Side Back -> LB
    if ((m & (1 << 4)) !== 0) bits |= INPUT_BUTTONS.RB; // Side Forward -> RB
    this._mouseButtonBits = bits >>> 0;
  }

  _clearMouseButtons() {
    this._mouseButtonsMask = 0;
    this._mouseButtonBits = 0;
    this._mouseLt = 0;
    this._mouseRt = 0;
  }

  _clearTransientInputs() {
    if (this.disposed) return;
    const hadKeys = this._pressedKeys.size > 0;
    const hadMouse = this._mouseButtonsMask !== 0;
    if (!hadKeys && !hadMouse) return;

    this._pressedKeys.clear();
    this._recomputeKeyboardState();
    this._clearMouseButtons();
    this._commitMergedState(true);
  }

  _selectGamepad() {
    if (!this._navigator || typeof this._navigator.getGamepads !== 'function') {
      return null;
    }
    let pads;
    try {
      pads = this._navigator.getGamepads();
    } catch {
      return null;
    }
    if (!pads) return null;

    if (typeof this._gamepadIndex === 'number') {
      const target = pads[this._gamepadIndex];
      return target && target.connected !== false ? target : null;
    }

    let fallback = null;
    for (let i = 0; i < pads.length; i++) {
      const pad = pads[i];
      if (!pad || pad.connected === false) continue;
      if (pad.mapping === 'standard') return pad;
      if (!fallback) fallback = pad;
    }
    return fallback;
  }

  /**
   * Polls navigator.getGamepads(), maps buttons/sticks/triggers with deadzones,
   * merges with keyboard + mouse state, and commits to the SharedArrayBuffer if changed.
   * Returns true if a new snapshot was committed.
   */
  pollGamepads() {
    if (this.disposed) return false;

    const pad = this._selectGamepad();
    if (!pad) {
      this._gpButtons = 0;
      this._gpLx = 0;
      this._gpLy = 0;
      this._gpRx = 0;
      this._gpRy = 0;
      this._gpLt = 0;
      this._gpRt = 0;
      return this._commitMergedState(false);
    }

    const buttons = pad.buttons || [];
    let gpMask = 0;
    const maxButtons = Math.min(buttons.length, STANDARD_GAMEPAD_BUTTON_MAP.length);
    for (let i = 0; i < maxButtons; i++) {
      const bit = STANDARD_GAMEPAD_BUTTON_MAP[i];
      if (bit !== 0 && isGamepadButtonPressed(buttons[i])) {
        gpMask |= bit;
      }
    }
    this._gpButtons = gpMask >>> 0;

    const rawLt = readGamepadTrigger(buttons[6]);
    const rawRt = readGamepadTrigger(buttons[7]);
    this._gpLt = applyTriggerDeadzone(rawLt, this._triggerDeadzone, this._triggerOuterDeadzone);
    this._gpRt = applyTriggerDeadzone(rawRt, this._triggerDeadzone, this._triggerOuterDeadzone);

    const axes = pad.axes || [];
    const ySign = this._invertGamepadY ? -1 : 1;
    const rawLx = Number.isFinite(axes[0]) ? axes[0] : 0;
    const rawLy = Number.isFinite(axes[1]) ? axes[1] * ySign : 0;
    const rawRx = Number.isFinite(axes[2]) ? axes[2] : 0;
    const rawRy = Number.isFinite(axes[3]) ? axes[3] * ySign : 0;

    const left = applyDeadzone(rawLx, rawLy, this._stickDeadzone, this._stickOuterDeadzone);
    const right = applyDeadzone(rawRx, rawRy, this._stickDeadzone, this._stickOuterDeadzone);

    this._gpLx = left.x;
    this._gpLy = left.y;
    this._gpRx = right.x;
    this._gpRy = right.y;

    return this._commitMergedState(false);
  }

  _commitMergedState(force = false) {
    const mergedButtons = (this._keyButtons | this._mouseButtonBits | this._gpButtons) >>> 0;

    // Gamepad axes override zero keyboard axes; keyboard WASD takes priority when active.
    const lx = clamp(this._keyLx !== 0 ? this._keyLx : this._gpLx, -1, 1);
    const ly = clamp(this._keyLy !== 0 ? this._keyLy : this._gpLy, -1, 1);
    const rx = clamp(this._gpRx, -1, 1);
    const ry = clamp(this._gpRy, -1, 1);

    const lt = clamp(Math.max(this._mouseLt, this._gpLt), 0, 1);
    const rt = clamp(Math.max(this._mouseRt, this._gpRt), 0, 1);

    // Compare IEEE-754 float32 representations so sub-f32 noise does not cause spurious writes.
    const lxBits = f32ToI32(lx);
    const lyBits = f32ToI32(ly);
    const rxBits = f32ToI32(rx);
    const ryBits = f32ToI32(ry);
    const ltBits = f32ToI32(lt);
    const rtBits = f32ToI32(rt);

    if (
      !force &&
      mergedButtons === this.buttons &&
      lxBits === f32ToI32(this._lastLx) &&
      lyBits === f32ToI32(this._lastLy) &&
      rxBits === f32ToI32(this._lastRx) &&
      ryBits === f32ToI32(this._lastRy) &&
      ltBits === f32ToI32(this._lastLt) &&
      rtBits === f32ToI32(this._lastRt)
    ) {
      return false;
    }

    this.buttons = mergedButtons;
    this._lastLx = i32ToF32(lxBits);
    this._lastLy = i32ToF32(lyBits);
    this._lastRx = i32ToF32(rxBits);
    this._lastRy = i32ToF32(ryBits);
    this._lastLt = i32ToF32(ltBits);
    this._lastRt = i32ToF32(rtBits);

    // Even/odd seqlock update:
    // 1. Bump seq to odd before writing fields.
    // 2. Write buttons and all 6 float32 channels atomically (preserving mouse_dx / mouse_dy).
    // 3. Bump seq to even after writing fields.
    Atomics.add(this.u32, 0, 1);
    Atomics.store(this.u32, 1, mergedButtons);
    Atomics.store(this.i32, 4, lxBits);
    Atomics.store(this.i32, 5, lyBits);
    Atomics.store(this.i32, 6, rxBits);
    Atomics.store(this.i32, 7, ryBits);
    Atomics.store(this.i32, 8, ltBits);
    Atomics.store(this.i32, 9, rtBits);
    Atomics.add(this.u32, 0, 1);

    return true;
  }

  _startLoop(isManual) {
    if (this.disposed) return;
    if (isManual) this._manualStart = true;
    if (this.running) return;
    if (!this._window || typeof this._window.requestAnimationFrame !== 'function') return;

    this.running = true;
    const tick = () => {
      if (!this.running || this.disposed) return;
      this.pollGamepads();
      if (this.running && !this.disposed) {
        this._rafId = this._window.requestAnimationFrame(tick);
      }
    };
    this._rafId = this._window.requestAnimationFrame(tick);
  }

  start() {
    this._startLoop(true);
  }

  stop() {
    this._manualStart = false;
    if (!this.running) return;
    this.running = false;
    if (
      this._rafId !== null &&
      this._window &&
      typeof this._window.cancelAnimationFrame === 'function'
    ) {
      this._window.cancelAnimationFrame(this._rafId);
    }
    this._rafId = null;
  }

  consumeWheelDelta() {
    const delta = this._wheelDelta;
    this._wheelDelta = 0;
    return delta;
  }

  reset() {
    this._pressedKeys.clear();
    this._recomputeKeyboardState();
    this._clearMouseButtons();
    this._wheelDelta = 0;

    this._gpButtons = 0;
    this._gpLx = 0;
    this._gpLy = 0;
    this._gpRx = 0;
    this._gpRy = 0;
    this._gpLt = 0;
    this._gpRt = 0;

    this.buttons = 0;
    this._lastLx = 0;
    this._lastLy = 0;
    this._lastRx = 0;
    this._lastRy = 0;
    this._lastLt = 0;
    this._lastRt = 0;

    Atomics.add(this.u32, 0, 1);
    Atomics.store(this.u32, 1, 0);
    Atomics.store(this.i32, 2, 0);
    Atomics.store(this.i32, 3, 0);
    Atomics.store(this.i32, 4, 0);
    Atomics.store(this.i32, 5, 0);
    Atomics.store(this.i32, 6, 0);
    Atomics.store(this.i32, 7, 0);
    Atomics.store(this.i32, 8, 0);
    Atomics.store(this.i32, 9, 0);
    Atomics.add(this.u32, 0, 1);
  }

  /**
   * Captures a consistent snapshot from the 40-byte SharedArrayBuffer using the same
   * even/odd seqlock protocol as playgta6::os::GameInputReader::read_snapshot().
   */
  getSnapshot(maxRetries = 64) {
    const attempts = maxRetries > 0 ? maxRetries : 1;
    for (let i = 0; i < attempts; i++) {
      const seqBefore = Atomics.load(this.u32, 0);
      if ((seqBefore & 1) !== 0) continue;

      const buttons = Atomics.load(this.u32, 1);
      const mouse_dx = Atomics.load(this.i32, 2);
      const mouse_dy = Atomics.load(this.i32, 3);
      const lx = i32ToF32(Atomics.load(this.i32, 4));
      const ly = i32ToF32(Atomics.load(this.i32, 5));
      const rx = i32ToF32(Atomics.load(this.i32, 6));
      const ry = i32ToF32(Atomics.load(this.i32, 7));
      const lt = i32ToF32(Atomics.load(this.i32, 8));
      const rt = i32ToF32(Atomics.load(this.i32, 9));

      const seqAfter = Atomics.load(this.u32, 0);
      if (seqBefore === seqAfter) {
        return {
          seq: seqBefore,
          buttons,
          mouse_dx,
          mouse_dy,
          lx,
          ly,
          rx,
          ry,
          lt,
          rt,
          pointerLocked: this.isPointerLocked(),
          wheelDelta: this._wheelDelta,
        };
      }
    }

    return null;
  }

  destroy() {
    if (this.disposed) return;
    this.stop();
    for (const { target, event, handler, options } of this._listeners) {
      if (target && typeof target.removeEventListener === 'function') {
        target.removeEventListener(event, handler, options);
      }
    }
    this._listeners.length = 0;
    this.reset();
    this.disposed = true;
  }

  dispose() {
    this.destroy();
  }
}
