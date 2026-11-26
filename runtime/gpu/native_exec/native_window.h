// Native executor: the optional host window the swapchain presents to (`--window`). Win32 only for
// milestone 0; other hosts get no window (create() fails) and run headless.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gpu/native_exec/window_layout.h"
#include "gpu/native_exec/window_script.h"

namespace playgta6::gpu::native_exec {

class NativeWindow {
public:
    // Creates and shows a resizable window with a client area of width x height. nullptr (and
    // *error) when the host has no window system support.
    static std::unique_ptr<NativeWindow> create(const std::string& title, uint32_t width, uint32_t height,
                                                std::string* error);
    virtual ~NativeWindow() = default;

    // Dispatches pending window messages (call from the thread that created the window). Returns
    // false once the user has closed the window.
    virtual bool pump() = 0;
    virtual bool closed() const = 0;
    // True while this is the foreground (keyboard-focused) window. Callable from any thread.
    virtual bool focused() const { return false; }
    // Sets the title bar text (UTF-8). Window thread only. Default: no-op.
    virtual void set_title(const std::string&) {}
    // True once after F12 was pressed (consumed by the call). Callable from any thread. Default: never.
    virtual bool take_screenshot_request() { return false; }
    // --show-target hotkeys: when enabled, F7 / F8 / F9 / F10 are swallowed and reported as bits
    // (show_target_model.h kHotkey*) by take_hotkeys(), which clears them. Window thread only.
    virtual void enable_hotkeys(bool) {}
    virtual uint32_t take_hotkeys() { return 0; }
    // Current client area size (0 x 0 while minimized).
    virtual void client_size(uint32_t* width, uint32_t* height) const = 0;
    // Where the executor should draw a src_w x src_h frame inside the client area: the largest
    // aspect-preserving rect, centred (letterbox / pillarbox). The executor clears the bars and
    // blits into this rect; empty while minimized.
    PresentRect present_rect(uint32_t src_w, uint32_t src_h) const {
        uint32_t w = 0, h = 0;
        client_size(&w, &h);
        return letterbox_rect(src_w, src_h, w, h);
    }
    // Test hook (--window-script): performs a resize / minimize / restore / maximize / fullscreen action
    // on the window thread. False when the host cannot (default). Probe is not a window action.
    virtual bool apply_action(const WindowAction&) { return false; }
    // Native handles for the WebGPU surface (Win32: HINSTANCE, HWND).
    virtual void* native_instance() const = 0;
    virtual void* native_handle() const = 0;
};

}  // namespace playgta6::gpu::native_exec
