// COM helpers, the ADVAPI32 registry, bcrypt (RNG only) and the USER32 window fake of the remaining
// Win32 core surface (task O15). See os/win32_compat.h for the overview.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "os/os_shim.h"
#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"
#include "os/win32_loader.h"

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;

// ---------------------------------------------------------------------------
// Randomness: std::random_device. On the host it reads the OS CSPRNG (getrandom / BCryptGenRandom /
// /dev/urandom). Under Emscripten libc++ opens /dev/urandom, which Emscripten backs with
// crypto.getRandomValues (browser) or node's crypto (node), so it is cryptographically strong
// there as well.
// ---------------------------------------------------------------------------

namespace playgta6::os::win32::compat_detail {

bool secure_random(void* dst, size_t size) {
    try {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        static std::random_device device;
        auto* out = static_cast<uint8_t*>(dst);
        size_t i = 0;
        while (i < size) {
            const uint32_t word = device();
            for (size_t b = 0; b < 4 && i < size; ++b, ++i) {
                out[i] = static_cast<uint8_t>(word >> (8 * b));
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace playgta6::os::win32::compat_detail

namespace {

// ---------------------------------------------------------------------------
// COM
// ---------------------------------------------------------------------------

thread_local int32_t t_com_mode = -1;  // -1 = not initialized, else COINIT apartment/mt bit
thread_local uint32_t t_com_count = 0;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

struct RegValue {
    uint32_t type = 0;
    std::vector<uint8_t> data;
};

struct Registry {
    std::mutex mutex;
    // normalized key path ("hklm\\software\\x") -> value name (folded) -> value
    std::map<std::string, std::map<std::u16string, RegValue>> keys;
};

Registry& registry() {
    static Registry r;
    return r;
}

struct RegKeyState {
    std::string path;
};

std::string lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c + 32);
        }
    }
    return s;
}

// "HKEY_LOCAL_MACHINE\\Software//X/" -> "hklm\\software\\x"
std::string normalize_key_path(std::string_view in) {
    std::string s(in);
    for (char& c : s) {
        if (c == '/') {
            c = '\\';
        }
    }
    std::string out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find('\\', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            if (!out.empty()) {
                out.push_back('\\');
            }
            out.append(s, pos, end - pos);
        }
        pos = end + 1;
    }
    out = lower_ascii(out);
    static const std::pair<const char*, const char*> kRoots[] = {
        {"hkey_classes_root", "hkcr"}, {"hkey_current_user", "hkcu"},
        {"hkey_local_machine", "hklm"}, {"hkey_users", "hku"}};
    for (const auto& r : kRoots) {
        const size_t n = std::strlen(r.first);
        if (out.compare(0, n, r.first) == 0 && (out.size() == n || out[n] == '\\')) {
            out = std::string(r.second) + out.substr(n);
            break;
        }
    }
    return out;
}

const char* predefined_root(uint64_t hkey) {
    switch (static_cast<uint32_t>(hkey)) {
        case 0x80000000u: return "hkcr";
        case 0x80000001u: return "hkcu";
        case 0x80000002u: return "hklm";
        case 0x80000003u: return "hku";
        default: return nullptr;
    }
}

bool is_predefined(uint64_t hkey) {
    return predefined_root(hkey) != nullptr &&
           (hkey >> 32 == 0 || hkey >> 32 == 0xFFFFFFFFull);
}

// Resolves an HKEY to its normalized path; false for invalid handles.
bool key_path_of(uint64_t hkey, std::string* path) {
    if (is_predefined(hkey)) {
        *path = predefined_root(hkey);
        return true;
    }
    auto state = std::static_pointer_cast<RegKeyState>(
        win32::lookup_opaque_object(hkey, cd::kTagRegKey));
    if (!state) {
        return false;
    }
    *path = state->path;
    return true;
}

// Called with the registry mutex held.
bool key_exists_locked(const Registry& r, const std::string& path) {
    if (path.find('\\') == std::string::npos) {
        return true;  // the four root keys always exist
    }
    if (r.keys.count(path) != 0) {
        return true;
    }
    const std::string prefix = path + "\\";
    auto it = r.keys.lower_bound(prefix);
    return it != r.keys.end() && it->first.compare(0, prefix.size(), prefix) == 0;
}

std::u16string fold_name(std::u16string_view s) {
    std::u16string out(s);
    for (char16_t& c : out) {
        c = cd::fold_char(c);
    }
    return out;
}

// ---------------------------------------------------------------------------
// bcrypt
// ---------------------------------------------------------------------------

struct AlgTable {
    std::mutex mutex;
    std::set<uint64_t> open;
    uint64_t next = 1;
};

AlgTable& algs() {
    static AlgTable t;
    return t;
}

constexpr uint64_t kAlgBase = 0x4243525900000000ULL;
constexpr int32_t kStatusUnsuccessful = static_cast<int32_t>(0xC0000001u);

// Case-insensitive comparison against a lowercase ASCII literal.
bool equals_ci(const char16_t* s, const char* lit) {
    size_t i = 0;
    for (; lit[i] != 0; ++i) {
        if (s[i] == 0 || cd::fold_char(s[i]) != static_cast<char16_t>(lit[i])) {
            return false;
        }
    }
    return s[i] == 0;
}

bool is_known_unsupported_algorithm(const char16_t* id) {
    static const char* kNames[] = {"aes", "sha1", "sha256", "sha384", "sha512", "md5", "md2",
                                   "md4", "rsa", "dsa", "dh", "3des", "des", "rc2", "rc4",
                                   "ecdsa_p256", "ecdsa_p384", "ecdsa_p521", "ecdh_p256",
                                   "ecdh_p384", "ecdh_p521", "pbkdf2", "hkdf", "dualec_rng",
                                   "fips186dsarng"};
    for (const char* n : kNames) {
        if (equals_ci(id, n)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// USER32 fake
// ---------------------------------------------------------------------------

struct WindowClass {
    uint16_t atom;
    uint64_t proc;
};

struct Window {
    std::string class_name;
    bool visible = false;
};

struct WindowState {
    std::mutex mutex;
    std::map<std::string, WindowClass> classes;  // folded class name
    std::map<uint64_t, Window> windows;
    std::deque<Pg6Msg> queue;
    uint16_t next_atom = 0xC000;
    uint64_t next_hwnd = 0x10000;
    uint64_t dispatched = 0;
};

WindowState& windows() {
    static WindowState s;
    return s;
}

constexpr size_t kMaxWindows = 4096;
constexpr size_t kMaxQueuedMessages = 10000;

}  // namespace

// ---------------------------------------------------------------------------
// Runtime-facing C++ API
// ---------------------------------------------------------------------------

namespace playgta6::os::win32 {

void registry_set_value(std::string_view key_path,
                        std::u16string_view value_name,
                        uint32_t type,
                        const void* data,
                        size_t size) {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    RegValue v;
    v.type = type;
    if (data != nullptr && size > 0) {
        v.data.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    }
    r.keys[normalize_key_path(key_path)][fold_name(value_name)] = std::move(v);
}

void registry_clear() {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.keys.clear();
}

bool post_window_message(Pg6Handle hwnd, uint32_t message, uint64_t wparam, int64_t lparam) {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    if ((hwnd != 0 && w.windows.count(hwnd) == 0) || w.queue.size() >= kMaxQueuedMessages) {
        return false;
    }
    Pg6Msg m{};
    m.hwnd = hwnd;
    m.message = message;
    m.wparam = wparam;
    m.lparam = lparam;
    m.time = static_cast<uint32_t>(GetTickCount64());
    w.queue.push_back(m);
    return true;
}

size_t window_count() {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    return w.windows.size();
}

size_t queued_message_count() {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    return w.queue.size();
}

uint64_t dispatched_message_count() {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    return w.dispatched;
}

void reset_window_state() {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    w.classes.clear();
    w.windows.clear();
    w.queue.clear();
    w.next_atom = 0xC000;
    w.next_hwnd = 0x10000;
    w.dispatched = 0;
}

}  // namespace playgta6::os::win32

extern "C" {

// ----- COM ------------------------------------------------------------------------------------

int32_t pg6_win32_co_initialize_ex(void* reserved, uint32_t co_init) {
    if (reserved != nullptr) {
        return win32::E_INVALIDARG;
    }
    const int32_t mode = (co_init & win32::COINIT_APARTMENTTHREADED) != 0 ? 1 : 0;
    if (t_com_mode >= 0 && t_com_mode != mode) {
        return win32::RPC_E_CHANGED_MODE;
    }
    const bool first = t_com_mode < 0;
    t_com_mode = mode;
    ++t_com_count;
    return first ? win32::S_OK : win32::S_FALSE;
}

void pg6_win32_co_uninitialize(void) {
    if (t_com_count > 0 && --t_com_count == 0) {
        t_com_mode = -1;
    }
}

int32_t pg6_win32_co_create_instance(const Pg6Guid* clsid,
                                     void*,
                                     uint32_t,
                                     const Pg6Guid*,
                                     void** object) {
    if (object == nullptr) {
        return win32::E_POINTER;
    }
    *object = nullptr;
    if (clsid == nullptr) {
        return win32::E_INVALIDARG;
    }
    // COM interfaces are task O16; log each distinct CLSID once so missing classes are visible.
    static std::mutex mutex;
    static std::set<std::string> seen;
    char text[48];
    std::snprintf(text, sizeof(text), "%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
                  clsid->data1, clsid->data2, clsid->data3, clsid->data4[0], clsid->data4[1],
                  clsid->data4[2], clsid->data4[3], clsid->data4[4], clsid->data4[5],
                  clsid->data4[6], clsid->data4[7]);
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        fresh = seen.insert(text).second;
    }
    if (fresh) {
        playgta6::os::log(std::string("CoCreateInstance: class not registered, CLSID {") + text + "}");
    }
    return win32::REGDB_E_CLASSNOTREG;
}

uint64_t pg6_win32_co_task_mem_alloc(uint64_t size) {
    return win32::HeapAlloc(win32::GetProcessHeap(), 0, size == 0 ? 1 : size);
}

void pg6_win32_co_task_mem_free(uint64_t memory) {
    if (memory != 0) {
        const uint32_t saved = win32::GetLastError();
        win32::HeapFree(win32::GetProcessHeap(), 0, memory);  // foreign pointers are rejected
        win32::SetLastError(saved);
    }
}

int32_t pg6_win32_co_create_guid(Pg6Guid* guid) {
    if (guid == nullptr) {
        return win32::E_INVALIDARG;
    }
    uint8_t raw[16];
    if (!cd::secure_random(raw, sizeof(raw))) {
        std::memset(guid, 0, sizeof(*guid));
        return static_cast<int32_t>(0x80004005u);  // E_FAIL
    }
    raw[6] = static_cast<uint8_t>((raw[6] & 0x0F) | 0x40);  // version 4
    raw[8] = static_cast<uint8_t>((raw[8] & 0x3F) | 0x80);  // RFC 4122 variant
    guid->data1 = static_cast<uint32_t>(raw[0]) << 24 | static_cast<uint32_t>(raw[1]) << 16 |
                  static_cast<uint32_t>(raw[2]) << 8 | raw[3];
    guid->data2 = static_cast<uint16_t>(raw[4] << 8 | raw[5]);
    guid->data3 = static_cast<uint16_t>(raw[6] << 8 | raw[7]);
    std::memcpy(guid->data4, raw + 8, 8);
    return win32::S_OK;
}

int32_t pg6_win32_string_from_guid2(const Pg6Guid* guid, char16_t* buffer, int32_t cch_max) {
    if (guid == nullptr || buffer == nullptr || cch_max < 39) {
        return 0;  // "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}" + NUL
    }
    char text[40];
    std::snprintf(text, sizeof(text), "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                  guid->data1, guid->data2, guid->data3, guid->data4[0], guid->data4[1],
                  guid->data4[2], guid->data4[3], guid->data4[4], guid->data4[5], guid->data4[6],
                  guid->data4[7]);
    for (int i = 0; i < 39; ++i) {
        buffer[i] = static_cast<char16_t>(static_cast<unsigned char>(text[i]));
    }
    return 39;
}

int32_t pg6_win32_prop_variant_clear(void* prop_variant) {
    if (prop_variant == nullptr) {
        return win32::E_INVALIDARG;
    }
    auto* bytes = static_cast<uint8_t*>(prop_variant);
    uint16_t vt = 0;
    std::memcpy(&vt, bytes, sizeof(vt));
    uint64_t pointer = 0;
    switch (vt) {
        case 30:  // VT_LPSTR
        case 31:  // VT_LPWSTR
        case 72:  // VT_CLSID
            std::memcpy(&pointer, bytes + 8, sizeof(pointer));
            break;
        case 65:  // VT_BLOB: {ULONG cbSize; BYTE* pBlobData}
            std::memcpy(&pointer, bytes + 16, sizeof(pointer));
            break;
        default:
            break;  // plain values; interface/BSTR members cannot be released here (task O16)
    }
    if (pointer != 0) {
        pg6_win32_co_task_mem_free(pointer);
    }
    std::memset(bytes, 0, 24);  // PROPVARIANT is 24 bytes on x64; vt = VT_EMPTY
    return win32::S_OK;
}

// ----- registry -------------------------------------------------------------------------------

int32_t pg6_win32_reg_open_key_w(Pg6Handle key, const char16_t* sub_key, Pg6Handle* result) {
    if (result == nullptr) {
        return static_cast<int32_t>(win32::ERROR_INVALID_PARAMETER);
    }
    *result = 0;
    std::string base;
    if (!key_path_of(key, &base)) {
        return static_cast<int32_t>(win32::ERROR_INVALID_HANDLE);
    }
    const std::string sub = sub_key == nullptr ? std::string() : win32::utf16_to_utf8(sub_key);
    std::string path = base;
    const std::string norm_sub = normalize_key_path(sub);
    if (!norm_sub.empty()) {
        path += "\\" + norm_sub;
    }
    {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        if (!key_exists_locked(r, path)) {
            return static_cast<int32_t>(win32::ERROR_FILE_NOT_FOUND);
        }
    }
    if (norm_sub.empty() && is_predefined(key)) {
        *result = key;  // opening a predefined key with no subkey returns it unchanged
        return 0;
    }
    auto state = std::make_shared<RegKeyState>();
    state->path = path;
    *result = win32::insert_opaque_object(cd::kTagRegKey, std::move(state));
    return 0;
}

int32_t pg6_win32_reg_query_value_ex_w(Pg6Handle key,
                                       const char16_t* value_name,
                                       uint32_t* reserved,
                                       uint32_t* type,
                                       uint8_t* data,
                                       uint32_t* data_size) {
    if (reserved != nullptr || (data != nullptr && data_size == nullptr)) {
        return static_cast<int32_t>(win32::ERROR_INVALID_PARAMETER);
    }
    std::string path;
    if (!key_path_of(key, &path)) {
        return static_cast<int32_t>(win32::ERROR_INVALID_HANDLE);
    }
    RegValue value;
    {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        auto kit = r.keys.find(path);
        if (kit == r.keys.end()) {
            return static_cast<int32_t>(win32::ERROR_FILE_NOT_FOUND);
        }
        const std::u16string name = fold_name(value_name == nullptr ? u"" : value_name);
        auto vit = kit->second.find(name);
        if (vit == kit->second.end()) {
            return static_cast<int32_t>(win32::ERROR_FILE_NOT_FOUND);
        }
        value = vit->second;
    }
    if (type != nullptr) {
        *type = value.type;
    }
    const uint32_t needed = static_cast<uint32_t>(value.data.size());
    if (data == nullptr) {
        if (data_size != nullptr) {
            *data_size = needed;
        }
        return 0;
    }
    if (*data_size < needed) {
        *data_size = needed;
        return static_cast<int32_t>(win32::ERROR_MORE_DATA);
    }
    if (needed > 0) {
        std::memcpy(data, value.data.data(), needed);
    }
    *data_size = needed;
    return 0;
}

int32_t pg6_win32_reg_close_key(Pg6Handle key) {
    if (is_predefined(key)) {
        return 0;
    }
    if (!win32::remove_opaque_object(key, cd::kTagRegKey)) {
        return static_cast<int32_t>(win32::ERROR_INVALID_HANDLE);
    }
    return 0;
}

// ----- bcrypt ---------------------------------------------------------------------------------

int32_t pg6_win32_bcrypt_open_algorithm_provider(Pg6Handle* algorithm,
                                                 const char16_t* algorithm_id,
                                                 const char16_t* implementation,
                                                 uint32_t flags) {
    if (algorithm == nullptr) {
        return win32::STATUS_INVALID_PARAMETER;
    }
    *algorithm = 0;
    if (algorithm_id == nullptr || flags != 0) {
        return win32::STATUS_INVALID_PARAMETER;
    }
    if (implementation != nullptr && implementation[0] != 0 &&
        !equals_ci(implementation, "microsoft primitive provider")) {
        return win32::STATUS_NOT_FOUND;
    }
    if (!equals_ci(algorithm_id, "rng")) {
        return is_known_unsupported_algorithm(algorithm_id) ? win32::STATUS_NOT_SUPPORTED
                                                            : win32::STATUS_NOT_FOUND;
    }
    auto& t = algs();
    std::lock_guard<std::mutex> lock(t.mutex);
    const uint64_t handle = kAlgBase + (t.next++ << 4);
    t.open.insert(handle);
    *algorithm = handle;
    return win32::STATUS_SUCCESS;
}

int32_t pg6_win32_bcrypt_gen_random(Pg6Handle algorithm,
                                    uint8_t* buffer,
                                    uint32_t byte_count,
                                    uint32_t flags) {
    constexpr uint32_t kUseEntropyInBuffer = 0x1;
    if ((flags & ~(kUseEntropyInBuffer | win32::BCRYPT_USE_SYSTEM_PREFERRED_RNG)) != 0) {
        return win32::STATUS_INVALID_PARAMETER;
    }
    if ((flags & win32::BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        if (algorithm != 0) {
            return win32::STATUS_INVALID_PARAMETER;
        }
    } else {
        auto& t = algs();
        std::lock_guard<std::mutex> lock(t.mutex);
        if (t.open.count(algorithm) == 0) {
            return win32::STATUS_INVALID_HANDLE;
        }
    }
    if (byte_count == 0) {
        return win32::STATUS_SUCCESS;
    }
    if (buffer == nullptr) {
        return win32::STATUS_INVALID_PARAMETER;
    }
    return cd::secure_random(buffer, byte_count) ? win32::STATUS_SUCCESS : kStatusUnsuccessful;
}

int32_t pg6_win32_bcrypt_close_algorithm_provider(Pg6Handle algorithm, uint32_t flags) {
    if (flags != 0) {
        return win32::STATUS_INVALID_PARAMETER;
    }
    auto& t = algs();
    std::lock_guard<std::mutex> lock(t.mutex);
    return t.open.erase(algorithm) != 0 ? win32::STATUS_SUCCESS : win32::STATUS_INVALID_HANDLE;
}

// ----- USER32 window fake ---------------------------------------------------------------------

uint16_t pg6_win32_register_class_ex_a(const Pg6WndClassExA* window_class) {
    if (window_class == nullptr || window_class->size != sizeof(Pg6WndClassExA) ||
        reinterpret_cast<uintptr_t>(window_class->class_name) < 0x10000 ||
        window_class->window_proc == 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    const std::string name = lower_ascii(window_class->class_name);
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    if (w.classes.count(name) != 0) {
        win32::SetLastError(win32::ERROR_CLASS_ALREADY_EXISTS);
        return 0;
    }
    const uint16_t atom = w.next_atom++;
    w.classes.emplace(name, WindowClass{atom, window_class->window_proc});
    win32::SetLastError(win32::ERROR_SUCCESS);
    return atom;
}

Pg6Handle pg6_win32_create_window_ex_a(uint32_t,
                                       const char* class_name,
                                       const char*,
                                       uint32_t,
                                       int32_t,
                                       int32_t,
                                       int32_t,
                                       int32_t,
                                       Pg6Handle parent,
                                       Pg6Handle,
                                       Pg6Handle,
                                       void*) {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    std::string folded;
    bool found = false;
    if (reinterpret_cast<uintptr_t>(class_name) < 0x10000 && class_name != nullptr) {
        const auto atom = static_cast<uint16_t>(reinterpret_cast<uintptr_t>(class_name));
        for (const auto& kv : w.classes) {
            if (kv.second.atom == atom) {
                folded = kv.first;
                found = true;
                break;
            }
        }
    } else if (class_name != nullptr) {
        folded = lower_ascii(class_name);
        found = w.classes.count(folded) != 0;
    }
    if (!found) {
        win32::SetLastError(win32::ERROR_CANNOT_FIND_WND_CLASS);
        return 0;
    }
    constexpr uint64_t kHwndMessage = ~uint64_t{2};  // HWND_MESSAGE
    if (parent != 0 && parent != kHwndMessage && w.windows.count(parent) == 0) {
        win32::SetLastError(win32::ERROR_INVALID_WINDOW_HANDLE);
        return 0;
    }
    if (w.windows.size() >= kMaxWindows) {
        win32::SetLastError(win32::ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    const uint64_t hwnd = w.next_hwnd;
    w.next_hwnd += 4;
    w.windows.emplace(hwnd, Window{folded, false});
    win32::SetLastError(win32::ERROR_SUCCESS);
    return hwnd;
}

int64_t pg6_win32_def_window_proc_a(Pg6Handle, uint32_t message, uint64_t, int64_t) {
    constexpr uint32_t kWmNcCreate = 0x0081;
    return message == kWmNcCreate ? 1 : 0;  // TRUE lets creation proceed; everything else is 0
}

int32_t pg6_win32_show_window(Pg6Handle hwnd, int32_t cmd_show) {
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    auto it = w.windows.find(hwnd);
    if (it == w.windows.end()) {
        win32::SetLastError(win32::ERROR_INVALID_WINDOW_HANDLE);
        return 0;
    }
    const bool was_visible = it->second.visible;
    it->second.visible = cmd_show != 0;  // SW_HIDE == 0
    win32::SetLastError(win32::ERROR_SUCCESS);
    return was_visible ? 1 : 0;
}

int32_t pg6_win32_peek_message_a(Pg6Msg* msg,
                                 Pg6Handle hwnd,
                                 uint32_t filter_min,
                                 uint32_t filter_max,
                                 uint32_t remove_message) {
    if (msg == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    const bool thread_only = hwnd == ~uint64_t{0};
    if (hwnd != 0 && !thread_only && w.windows.count(hwnd) == 0) {
        win32::SetLastError(win32::ERROR_INVALID_WINDOW_HANDLE);
        return 0;
    }
    const bool any_range = filter_min == 0 && filter_max == 0;
    win32::SetLastError(win32::ERROR_SUCCESS);
    for (auto it = w.queue.begin(); it != w.queue.end(); ++it) {
        if (thread_only ? it->hwnd != 0 : (hwnd != 0 && it->hwnd != hwnd)) {
            continue;
        }
        if (!any_range && (it->message < filter_min || it->message > filter_max)) {
            continue;
        }
        *msg = *it;
        if ((remove_message & win32::PM_REMOVE) != 0) {
            w.queue.erase(it);
        }
        return 1;
    }
    return 0;
}

int32_t pg6_win32_translate_message(const Pg6Msg* msg) {
    if (msg == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    return 0;  // no WM_CHAR is ever generated, so nothing is translated
}

int64_t pg6_win32_dispatch_message_a(const Pg6Msg* msg) {
    if (msg == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // The window procedure is a guest function; calling it needs the O16 trampolines. The message
    // is counted and treated as handled with result 0.
    auto& w = windows();
    std::lock_guard<std::mutex> lock(w.mutex);
    ++w.dispatched;
    return 0;
}

}  // extern "C"
