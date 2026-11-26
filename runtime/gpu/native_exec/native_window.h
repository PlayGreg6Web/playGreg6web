// Native executor: the optional host window the swapchain presents to (`--window`). Win32 only for
// milestone 0; other hosts get no window (create() fails) and run headless.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

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
    // Current client area size (0 x 0 while minimized).
    virtual void client_size(uint32_t* width, uint32_t* height) const = 0;
    // Native handles for the WebGPU surface (Win32: HINSTANCE, HWND).
    virtual void* native_instance() const = 0;
    virtual void* native_handle() const = 0;
};

}  // namespace playgta6::gpu::native_exec
