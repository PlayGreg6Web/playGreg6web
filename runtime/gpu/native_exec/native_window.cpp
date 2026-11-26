#include "gpu/native_exec/native_window.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace playgta6::gpu::native_exec {

#ifdef _WIN32
namespace {

constexpr wchar_t kClassName[] = L"playgta6_native_exec";

class Win32Window final : public NativeWindow {
public:
    ~Win32Window() override {
        if (hwnd_) DestroyWindow(hwnd_);
    }

    bool init(const std::string& title, uint32_t width, uint32_t height, std::string* error) {
        instance_ = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof wc;
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &Win32Window::proc;
        wc.hInstance = instance_;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
        wc.lpszClassName = kClassName;
        if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            if (error) *error = "RegisterClassExW failed";
            return false;
        }
        RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        const DWORD style = WS_OVERLAPPEDWINDOW;
        AdjustWindowRect(&rect, style, FALSE);
        std::wstring wtitle(title.begin(), title.end());
        hwnd_ = CreateWindowExW(0, kClassName, wtitle.c_str(), style, CW_USEDEFAULT, CW_USEDEFAULT,
                                rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, instance_, this);
        if (!hwnd_) {
            if (error) *error = "CreateWindowExW failed";
            return false;
        }
        ShowWindow(hwnd_, SW_SHOWNORMAL);
        UpdateWindow(hwnd_);
        return true;
    }

    bool pump() override {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        return !closed_;
    }
    bool closed() const override { return closed_; }
    bool focused() const override { return hwnd_ != nullptr && !closed_ && GetForegroundWindow() == hwnd_; }
    void client_size(uint32_t* width, uint32_t* height) const override {
        RECT r{};
        if (hwnd_ && GetClientRect(hwnd_, &r)) {
            *width = static_cast<uint32_t>(r.right - r.left);
            *height = static_cast<uint32_t>(r.bottom - r.top);
        } else {
            *width = *height = 0;
        }
    }
    void* native_instance() const override { return instance_; }
    void* native_handle() const override { return hwnd_; }

private:
    static LRESULT CALLBACK proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        if (msg == WM_NCCREATE) {
            const auto* cs = reinterpret_cast<const CREATESTRUCTW*>(lp);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        }
        auto* self = reinterpret_cast<Win32Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self && msg == WM_CLOSE) {
            // Keep the HWND alive (the surface still references it); the owner stops presenting.
            self->closed_ = true;
            ShowWindow(hwnd, SW_HIDE);
            return 0;
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    HINSTANCE instance_ = nullptr;
    HWND hwnd_ = nullptr;
    bool closed_ = false;
};

}  // namespace

std::unique_ptr<NativeWindow> NativeWindow::create(const std::string& title, uint32_t width, uint32_t height,
                                                   std::string* error) {
    auto w = std::make_unique<Win32Window>();
    if (!w->init(title, width, height, error)) return nullptr;
    return w;
}
#else
std::unique_ptr<NativeWindow> NativeWindow::create(const std::string&, uint32_t, uint32_t, std::string* error) {
    if (error) *error = "native window: only Win32 is implemented (milestone 0); run headless";
    return nullptr;
}
#endif

}  // namespace playgta6::gpu::native_exec
