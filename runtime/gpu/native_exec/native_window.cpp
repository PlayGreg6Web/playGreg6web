#include "gpu/native_exec/native_window.h"

#include "gpu/native_exec/window_layout.h"
#include "os/raw_mouse.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <atomic>
#include <windows.h>

#include <fstream>
#include <sstream>
#include <vector>
#endif

namespace playgta6::gpu::native_exec {

#ifdef _WIN32
namespace {

constexpr wchar_t kClassName[] = L"playgta6_native_exec";


using SetDpiCtxFn = BOOL(WINAPI*)(HANDLE);
using GetDpiForWindowFn = UINT(WINAPI*)(HWND);
using AdjustForDpiFn = BOOL(WINAPI*)(LPRECT, DWORD, BOOL, DWORD, UINT);

template <class F>
F user32_fn(const char* name) {
    HMODULE m = GetModuleHandleW(L"user32.dll");
    return m ? reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(m, name))) : nullptr;
}

// Per-monitor-v2 DPI awareness (Win10 1703+); older hosts stay unaware, which is still correct
// (the system scales the window). Must run before the first window is created.
void enable_dpi_awareness() {
    static bool done = false;
    if (done) return;
    done = true;
    if (auto fn = user32_fn<SetDpiCtxFn>("SetProcessDpiAwarenessContext")) {
        fn(reinterpret_cast<HANDLE>(static_cast<intptr_t>(-4)));  // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    }
}

UINT dpi_of(HWND hwnd) {
    if (auto fn = user32_fn<GetDpiForWindowFn>("GetDpiForWindow")) {
        const UINT d = fn(hwnd);
        if (d) return d;
    }
    return 96;
}

void adjust_rect_for_dpi(RECT* r, DWORD style, UINT dpi) {
    if (auto fn = user32_fn<AdjustForDpiFn>("AdjustWindowRectExForDpi")) {
        if (fn(r, style, FALSE, 0, dpi)) return;
    }
    AdjustWindowRect(r, style, FALSE);
}

std::wstring state_path() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD cap = static_cast<DWORD>(sizeof buf / sizeof buf[0]);
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, cap);
    if (n == 0 || n >= cap) return {};
    return std::wstring(buf) + L"\\playgta6\\native_window.ini";
}

bool load_state(SavedWindowState* out) {
    const std::wstring p = state_path();
    if (p.empty()) return false;
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_window_state(ss.str(), out);
}

void save_state(const SavedWindowState& st) {
    const std::wstring p = state_path();
    if (p.empty()) return;
    CreateDirectoryW(p.substr(0, p.find_last_of(L'\\')).c_str(), nullptr);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (f) f << format_window_state(st);
}

BOOL CALLBACK collect_monitor(HMONITOR mon, HDC, LPRECT, LPARAM lp) {
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    if (GetMonitorInfoW(mon, &mi)) {
        reinterpret_cast<std::vector<MonitorRect>*>(lp)->push_back(
            {mi.rcWork.left, mi.rcWork.top, mi.rcWork.right, mi.rcWork.bottom});
    }
    return TRUE;
}

class Win32Window final : public NativeWindow {
public:
    ~Win32Window() override {
        if (!closed_) save_placement();
        release_capture();
        playgta6::os::raw_mouse::state().window_ready.store(false);
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
        enable_dpi_awareness();
        // Restore the last size/position when it still fits the current monitors; else the requested
        // client size (a 96-dpi length, scaled to the primary monitor's dpi) at the default spot.
        const DWORD style = WS_OVERLAPPEDWINDOW;
        int x = CW_USEDEFAULT, y = CW_USEDEFAULT;
        UINT dpi = 96;
        if (HDC dc = GetDC(nullptr)) {
            dpi = static_cast<UINT>(GetDeviceCaps(dc, LOGPIXELSX));
            ReleaseDC(nullptr, dc);
        }
        RECT rect{0, 0, scale_for_dpi(static_cast<int32_t>(width), dpi),
                  scale_for_dpi(static_cast<int32_t>(height), dpi)};
        adjust_rect_for_dpi(&rect, style, dpi);
        int outer_w = rect.right - rect.left, outer_h = rect.bottom - rect.top;
        SavedWindowState saved;
        std::vector<MonitorRect> monitors;
        EnumDisplayMonitors(nullptr, nullptr, &collect_monitor, reinterpret_cast<LPARAM>(&monitors));
        if (load_state(&saved) && fit_state_to_monitors(&saved, monitors)) {
            x = saved.x;
            y = saved.y;
            outer_w = static_cast<int>(saved.w);
            outer_h = static_cast<int>(saved.h);
            start_maximized_ = saved.maximized;
        }
        std::wstring wtitle(title.begin(), title.end());
        hwnd_ = CreateWindowExW(0, kClassName, wtitle.c_str(), style, x, y, outer_w, outer_h, nullptr, nullptr,
                                instance_, this);
        if (!hwnd_) {
            if (error) *error = "CreateWindowExW failed";
            return false;
        }
        ShowWindow(hwnd_, start_maximized_ ? SW_SHOWMAXIMIZED : SW_SHOWNORMAL);
        // Raw mouse input for mouse look. No RIDEV_INPUTSINK: WM_INPUT only arrives while this window is
        // in the foreground. Deltas are only kept while the cursor is captured (see update_capture()).
        RAWINPUTDEVICE rid{};
        rid.usUsagePage = 0x01;  // generic desktop
        rid.usUsage = 0x02;      // mouse
        rid.dwFlags = 0;
        rid.hwndTarget = hwnd_;
        if (RegisterRawInputDevices(&rid, 1, sizeof rid)) {
            playgta6::os::raw_mouse::state().window_ready.store(true);
        }
        UpdateWindow(hwnd_);
        return true;
    }

    bool pump() override {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        // Mouse look may be switched on (keymap loaded) after the window already has focus.
        if (!capturing_ && !capture_released_ && focus_ && playgta6::os::raw_mouse::state().look_enabled.load()) {
            update_capture();
        }
        return !closed_;
    }
    bool closed() const override { return closed_; }
    bool take_screenshot_request() override { return shot_request_.exchange(false); }
    void set_title(const std::string& title) override {
        if (!hwnd_ || closed_) return;
        const int n = MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, nullptr, 0);
        if (n <= 0) return;
        std::wstring w(static_cast<size_t>(n), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, w.data(), n);
        SetWindowTextW(hwnd_, w.c_str());
    }
    // Keyboard focus, not just foreground: a minimized window reports false so --host-input
    // releases held keys instead of leaving them stuck.
    void enable_hotkeys(bool on) override { hotkeys_on_ = on; }
    uint32_t take_hotkeys() override {
        const uint32_t r = hotkeys_pending_;
        hotkeys_pending_ = 0;
        return r;
    }
    bool focused() const override {
        return hwnd_ != nullptr && !closed_ && !IsIconic(hwnd_) && GetForegroundWindow() == hwnd_;
    }
    void client_size(uint32_t* width, uint32_t* height) const override {
        RECT r{};
        if (hwnd_ && GetClientRect(hwnd_, &r)) {
            *width = static_cast<uint32_t>(r.right - r.left);
            *height = static_cast<uint32_t>(r.bottom - r.top);
        } else {
            *width = *height = 0;
        }
    }
    bool apply_action(const WindowAction& a) override {
        if (!hwnd_ || closed_) return false;
        switch (a.kind) {
            case WindowActionKind::Resize: {
                if (fullscreen_) toggle_fullscreen();
                if (IsIconic(hwnd_) || IsZoomed(hwnd_)) ShowWindow(hwnd_, SW_RESTORE);
                const UINT dpi = dpi_of(hwnd_);
                RECT r{0, 0, static_cast<LONG>(a.w), static_cast<LONG>(a.h)};  // physical client pixels
                adjust_rect_for_dpi(&r, WS_OVERLAPPEDWINDOW, dpi);
                return SetWindowPos(hwnd_, nullptr, 0, 0, r.right - r.left, r.bottom - r.top,
                                    SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE) != 0;
            }
            case WindowActionKind::Minimize:
                ShowWindow(hwnd_, SW_MINIMIZE);
                return true;
            case WindowActionKind::Restore:
                ShowWindow(hwnd_, SW_RESTORE);
                return true;
            case WindowActionKind::Maximize:
                if (fullscreen_) toggle_fullscreen();
                ShowWindow(hwnd_, SW_MAXIMIZE);
                return true;
            case WindowActionKind::Fullscreen:
                if (IsIconic(hwnd_)) ShowWindow(hwnd_, SW_RESTORE);
                if (!fullscreen_) toggle_fullscreen();
                return fullscreen_;
            case WindowActionKind::Windowed:
                if (fullscreen_) toggle_fullscreen();
                return !fullscreen_;
            case WindowActionKind::Probe:
                break;
        }
        return false;
    }
    void* native_instance() const override { return instance_; }
    void* native_handle() const override { return hwnd_; }

private:
    // Saves the restored (non-maximized, non-fullscreen) outer rect for the next run.
    void save_placement() {
        if (!hwnd_) return;
        WINDOWPLACEMENT wp{};
        wp.length = sizeof wp;
        if (fullscreen_) wp = placement_;
        else if (!GetWindowPlacement(hwnd_, &wp)) return;
        const RECT& r = wp.rcNormalPosition;
        SavedWindowState st;
        st.x = r.left;
        st.y = r.top;
        st.w = static_cast<uint32_t>(r.right > r.left ? r.right - r.left : 0);
        st.h = static_cast<uint32_t>(r.bottom > r.top ? r.bottom - r.top : 0);
        st.maximized = wp.showCmd == SW_SHOWMAXIMIZED;
        if (st.w < 16 || st.h < 16) return;
        save_state(st);
    }

    // Mouse look: while the window has focus, is not minimised, mouse look is on and the user has not
    // released it, the cursor is hidden and confined to the client rect and Raw Input deltas are kept.
    void update_capture() {
        namespace rm = playgta6::os::raw_mouse;
        const bool want = hwnd_ != nullptr && !closed_ && focus_ && !IsIconic(hwnd_) && !capture_released_ &&
                          rm::state().look_enabled.load();
        if (want) {
            RECT r{};
            POINT tl{0, 0};
            if (GetClientRect(hwnd_, &r) && ClientToScreen(hwnd_, &tl)) {
                RECT clip{tl.x, tl.y, tl.x + r.right, tl.y + r.bottom};
                if (r.right > 0 && r.bottom > 0) ClipCursor(&clip);
            }
        } else if (capturing_) {
            ClipCursor(nullptr);
        }
        capturing_ = want;
        rm::state().capturing.store(want);
        if (!want) {
            int32_t dx, dy;
            rm::take(&dx, &dy);
        }
        update_cursor();
    }
    void release_capture() {
        capture_released_ = true;
        update_capture();
    }

    // Fullscreen or mouse capture hides the cursor while the window has focus; windowed mode keeps the arrow.
    void update_cursor() {
        if ((fullscreen_ || capturing_) && focus_) {
            if (!cursor_hidden_) {
                while (ShowCursor(FALSE) >= 0) {}
                cursor_hidden_ = true;
            }
        } else if (cursor_hidden_) {
            while (ShowCursor(TRUE) < 0) {}
            cursor_hidden_ = false;
        }
    }

    void toggle_fullscreen() {
        const LONG_PTR style = GetWindowLongPtrW(hwnd_, GWL_STYLE);
        if (!fullscreen_) {
            MONITORINFO mi{};
            mi.cbSize = sizeof mi;
            placement_.length = sizeof placement_;
            if (!GetWindowPlacement(hwnd_, &placement_) ||
                !GetMonitorInfoW(MonitorFromWindow(hwnd_, MONITOR_DEFAULTTONEAREST), &mi)) {
                return;
            }
            SetWindowLongPtrW(hwnd_, GWL_STYLE, style & ~static_cast<LONG_PTR>(WS_OVERLAPPEDWINDOW));
            SetWindowPos(hwnd_, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            fullscreen_ = true;
        } else {
            SetWindowLongPtrW(hwnd_, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(hwnd_, &placement_);
            SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            fullscreen_ = false;
        }
        update_cursor();
    }

    static LRESULT CALLBACK proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        if (msg == WM_NCCREATE) {
            const auto* cs = reinterpret_cast<const CREATESTRUCTW*>(lp);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        }
        auto* self = reinterpret_cast<Win32Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        // F11 toggles borderless fullscreen. Esc (and every other key) is left alone: the title's
        // Menu button reads it through the host input.
        if (self && msg == WM_KEYDOWN && wp == VK_F11 && (lp & (1 << 30)) == 0) {
            self->toggle_fullscreen();
            return 0;
        }
        // --show-target hotkeys (F7 mip, F8 layer, F9 previous, F10 next target). F10 arrives as WM_SYSKEYDOWN.
        if (self && self->hotkeys_on_ && (msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN) && (lp & (1 << 30)) == 0) {
            const uint32_t bit = wp == VK_F7 ? 1u : wp == VK_F8 ? 2u : wp == VK_F9 ? 4u : wp == VK_F10 ? 8u : 0u;
            if (bit) {
                self->hotkeys_pending_ |= bit;
                return 0;
            }
        }
        if (self && msg == WM_KEYDOWN && wp == VK_F12 && (lp & (1 << 30)) == 0) {
            self->shot_request_ = true;  // the executor saves the next presented frame
            return 0;
        }
        if (self && (msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN) && (lp & (1 << 30)) == 0 &&
            static_cast<int>(wp) == playgta6::os::raw_mouse::state().release_vk.load() &&
            playgta6::os::raw_mouse::state().look_enabled.load()) {
            // Configurable release key (keymap.ini [mouse] release_key) toggles the cursor capture.
            self->capture_released_ = !self->capture_released_;
            self->update_capture();
            if (msg == WM_SYSKEYDOWN) return 0;  // keep Alt from opening the system menu
        }
        if (self) {
            switch (msg) {
                case WM_SETFOCUS:
                    self->focus_ = true;
                    self->update_capture();
                    break;
                case WM_KILLFOCUS:
                    self->focus_ = false;
                    self->update_capture();
                    break;
                case WM_SIZE:
                case WM_MOVE:
                case WM_SHOWWINDOW:
                    self->update_capture();  // minimise releases, resize/move re-fits the clip rect
                    break;
                case WM_LBUTTONDOWN:
                    if (self->capture_released_) {
                        self->capture_released_ = false;
                        self->update_capture();
                    }
                    break;
                case WM_INPUT: {
                    RAWINPUT in{};
                    UINT size = sizeof in;
                    if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lp), RID_INPUT, &in, &size,
                                        sizeof(RAWINPUTHEADER)) != static_cast<UINT>(-1) &&
                        in.header.dwType == RIM_TYPEMOUSE && (in.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE) == 0) {
                        playgta6::os::raw_mouse::add(in.data.mouse.lLastX, in.data.mouse.lLastY);
                    }
                    break;  // DefWindowProc must still run for cleanup
                }
                case WM_ERASEBKGND:
                    return 1;  // the swapchain covers the client area; no GDI flicker on resize
                case WM_GETMINMAXINFO: {
                    auto* mmi = reinterpret_cast<MINMAXINFO*>(lp);
                    const UINT dpi = dpi_of(hwnd);
                    RECT r{0, 0, scale_for_dpi(320, dpi), scale_for_dpi(180, dpi)};
                    adjust_rect_for_dpi(&r, WS_OVERLAPPEDWINDOW, dpi);
                    mmi->ptMinTrackSize.x = r.right - r.left;
                    mmi->ptMinTrackSize.y = r.bottom - r.top;
                    return 0;
                }
                case WM_DPICHANGED: {
                    // Move to the rect Windows suggests so the window keeps its logical size on the
                    // new monitor; the swapchain follows through client_size().
                    const RECT* sug = reinterpret_cast<const RECT*>(lp);
                    if (!self->fullscreen_) {
                        SetWindowPos(hwnd, nullptr, sug->left, sug->top, sug->right - sug->left,
                                     sug->bottom - sug->top, SWP_NOZORDER | SWP_NOACTIVATE);
                    }
                    return 0;
                }
                case WM_SETCURSOR:
                    if (LOWORD(lp) == HTCLIENT && (self->fullscreen_ || self->capturing_) && self->focus_) {
                        SetCursor(nullptr);
                        return TRUE;
                    }
                    break;
                default:
                    break;
            }
        }
        if (self && msg == WM_CLOSE) {
            self->save_placement();
            self->capture_released_ = true;
            self->update_capture();
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
    bool fullscreen_ = false;
    bool start_maximized_ = false;
    bool focus_ = false;
    std::atomic<bool> shot_request_{false};
    bool hotkeys_on_ = false;
    uint32_t hotkeys_pending_ = 0;
    bool cursor_hidden_ = false;
    bool capturing_ = false;
    bool capture_released_ = false;
    WINDOWPLACEMENT placement_{};
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
