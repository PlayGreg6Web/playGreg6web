#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include <thread>

#include "guest/boot.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/image_loader.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_sync.h"

// Boot sequence tests: a synthetic PGIM image (PE headers with a TLS directory, IAT, TLS template)
// and hand-written guest-convention functions standing in for lifted code. Like lifted code they
// keep the guest stack in lockstep (push a return address before a call, pop it on `ret`), but
// they access guest memory through the arena instead of 1:1 pointers.

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_boot.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
using playgta6::os::VirtualMemoryArena;

// ---------------------------------------------------------------------------
// Minimal JSON reader (enough to parse the report back)
// ---------------------------------------------------------------------------

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool boolean = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json& operator[](const std::string& key) const {
        static const Json kNull;
        for (const auto& [k, v] : fields) {
            if (k == key) {
                return v;
            }
        }
        return kNull;
    }
    const Json& operator[](size_t i) const {
        static const Json kNull;
        return i < items.size() ? items[i] : kNull;
    }
    bool has(const std::string& key) const {
        for (const auto& f : fields) {
            if (f.first == key) {
                return true;
            }
        }
        return false;
    }
    uint64_t hex() const { return type == String ? std::strtoull(text.c_str(), nullptr, 16) : 0; }
};

struct JsonParser {
    const std::string& s;
    size_t i = 0;
    bool ok = true;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    bool eat(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }
    std::string str() {
        std::string out;
        if (!eat('"')) {
            ok = false;
            return out;
        }
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                const char e = s[i];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'r') out += '\r';
                else if (e == 'u') {
                    out += static_cast<char>(std::strtol(s.substr(i + 1, 4).c_str(), nullptr, 16));
                    i += 4;
                } else out += e;
                ++i;
            } else {
                out += s[i++];
            }
        }
        if (i >= s.size()) ok = false;
        ++i;
        return out;
    }
    Json value() {
        Json v;
        ws();
        if (i >= s.size()) {
            ok = false;
            return v;
        }
        const char c = s[i];
        if (c == '{') {
            ++i;
            v.type = Json::Object;
            if (eat('}')) return v;
            do {
                std::string key = str();
                if (!eat(':')) ok = false;
                v.fields.emplace_back(key, value());
            } while (ok && eat(','));
            if (!eat('}')) ok = false;
        } else if (c == '[') {
            ++i;
            v.type = Json::Array;
            if (eat(']')) return v;
            do {
                v.items.push_back(value());
            } while (ok && eat(','));
            if (!eat(']')) ok = false;
        } else if (c == '"') {
            v.type = Json::String;
            v.text = str();
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (s.compare(i, 4, "true") == 0) {
            v.type = Json::Bool;
            v.boolean = true;
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            v.type = Json::Bool;
            i += 5;
        } else {
            v.type = Json::Number;
            char* end = nullptr;
            v.number = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) ok = false;
            i = static_cast<size_t>(end - s.c_str());
        }
        return v;
    }
};

Json parse_json(const std::string& text, bool* ok) {
    JsonParser p{text};
    Json v = p.value();
    p.ws();
    *ok = p.ok && p.i == text.size();
    return v;
}

// ---------------------------------------------------------------------------
// Synthetic image
// ---------------------------------------------------------------------------

constexpr uint64_t kBase = 0x140000000ULL;
constexpr uint32_t kSizeOfImage = 0x6000;
constexpr uint32_t kEntryRva = 0x1000;
constexpr uint64_t kEntry = kBase + kEntryRva;
constexpr uint64_t kMid = kBase + 0x1100;
constexpr uint64_t kLeaf = kBase + 0x1180;
constexpr uint64_t kCallback1 = kBase + 0x1200;
constexpr uint64_t kCallback2 = kBase + 0x1240;
constexpr uint64_t kWorker = kBase + 0x1300;
// Return addresses inside the functions above, each preceded by a call instruction in .text.
constexpr uint64_t kEntryCallsMid = kBase + 0x1010;   // after `call rel32` at 0x100B
constexpr uint64_t kMidCallsLeaf = kBase + 0x1120;    // after `call [rip+disp]` at 0x111A
constexpr uint64_t kEntryCallsIat = kBase + 0x1030;   // after `call [rip+disp]` at 0x102A
constexpr uint64_t kEntryStaleValue = kBase + 0x1050;  // inside entry, NOT after a call
constexpr uint64_t kUnknownInsideEntry = kBase + 0x1234 - 0x200;  // 0x1034: not an entry

constexpr uint32_t kTlsDirRva = 0x2000;
constexpr uint32_t kCallbacksRva = 0x2040;
constexpr uint32_t kIatRva = 0x3000;
constexpr uint32_t kTlsIndexRva = 0x3100;
constexpr uint32_t kTlsTemplateRva = 0x4000;
constexpr uint32_t kTlsTemplateSize = 0x40;
constexpr uint32_t kTlsZeroFill = 0x30;
constexpr uint32_t kLfanew = 0x80;
constexpr uint64_t kPeStackReserve = 0x40000;

// IAT slots.
constexpr uint32_t kSlotExit = kIatRva + 0x00;
constexpr uint32_t kSlotSetLastError = kIatRva + 0x08;
constexpr uint32_t kSlotGetLastError = kIatRva + 0x10;
constexpr uint32_t kSlotMissingOne = kIatRva + 0x18;
constexpr uint32_t kSlotMissingTwo = kIatRva + 0x20;
constexpr uint32_t kSlotAbort = kIatRva + 0x28;

void put32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void set16(std::vector<uint8_t>& b, size_t at, uint16_t v) {
    b[at] = static_cast<uint8_t>(v);
    b[at + 1] = static_cast<uint8_t>(v >> 8);
}
void set32(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}
void set64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

struct Sec {
    std::string name;
    uint32_t rva;
    uint32_t vsize;
    uint32_t prot;
    uint32_t flags;
    std::vector<uint8_t> data;
};

std::vector<uint8_t> write_container(const std::vector<Sec>& secs, uint32_t flags, uint64_t base) {
    const uint64_t tables = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables;
    for (const Sec& s : secs) {
        if (s.data.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.data.size();
    }
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put32(out, 1);
    put32(out, 64);
    put32(out, 64);
    put32(out, flags);
    put32(out, 4096);
    put64(out, base);
    put32(out, kSizeOfImage);
    put32(out, 0x1000);
    put32(out, kEntryRva);
    put32(out, static_cast<uint32_t>(secs.size()));
    put32(out, 0);
    put32(out, 0);
    put64(out, cursor);
    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put32(out, secs[i].rva);
        put32(out, secs[i].vsize);
        put32(out, secs[i].prot);
        put32(out, secs[i].flags);
        put64(out, offsets[i]);
        put64(out, secs[i].data.size());
        put64(out, os::pgim::fnv1a_64(secs[i].data.data(), secs[i].data.size()));
        put32(out, 0);
        put32(out, 0);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        if (!secs[i].data.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), secs[i].data.begin(), secs[i].data.end());
        }
    }
    return out;
}

uint8_t template_byte(size_t i) { return static_cast<uint8_t>(0xA0 + i); }

// with_tls = false drops the TLS data directory.
std::vector<uint8_t> build_image(bool with_tls = true, uint64_t base = kBase,
                                 std::vector<uint64_t> callbacks = {kCallback1, kCallback2},
                                 uint8_t seed = 0xA0) {
    using namespace os::pgim;
    // PE headers: DOS stub + PE32+ optional header with 16 data directories.
    std::vector<uint8_t> headers(0x400, 0);
    set16(headers, 0, 0x5A4D);
    set32(headers, 0x3C, kLfanew);
    set32(headers, kLfanew, 0x00004550);
    set16(headers, kLfanew + 4, 0x8664);  // machine
    set16(headers, kLfanew + 20, 240);    // SizeOfOptionalHeader
    const size_t opt = kLfanew + 24;
    set16(headers, opt, 0x20B);
    set32(headers, opt + 16, kEntryRva);
    set64(headers, opt + 24, base);
    set16(headers, opt + 68, 2);  // IMAGE_SUBSYSTEM_WINDOWS_GUI
    set64(headers, opt + 72, kPeStackReserve);
    set32(headers, opt + 108, 16);
    if (with_tls) {
        set32(headers, opt + 112 + 8 * 9, kTlsDirRva);
        set32(headers, opt + 112 + 8 * 9 + 4, 40);
    }

    // .text: int3 filler with call instructions right before each return address used below.
    std::vector<uint8_t> text(0x400, 0xCC);
    auto at = [](uint64_t va) { return static_cast<size_t>(va - kBase - 0x1000); };
    text[at(kEntryCallsMid) - 5] = 0xE8;  // call rel32
    text[at(kMidCallsLeaf) - 6] = 0xFF;   // call [rip+disp32]
    text[at(kMidCallsLeaf) - 5] = 0x15;
    text[at(kEntryCallsIat) - 6] = 0xFF;
    text[at(kEntryCallsIat) - 5] = 0x15;

    // .rdata: TLS directory + callback array.
    std::vector<uint8_t> rdata(0x100, 0);
    set64(rdata, 0x00, base + kTlsTemplateRva);
    set64(rdata, 0x08, base + kTlsTemplateRva + kTlsTemplateSize);
    set64(rdata, 0x10, base + kTlsIndexRva);
    set64(rdata, 0x18, base + kCallbacksRva);
    set32(rdata, 0x20, kTlsZeroFill);
    set32(rdata, 0x24, 0x00500000);  // IMAGE_SCN_ALIGN_16BYTES
    for (size_t i = 0; i < callbacks.size(); ++i) set64(rdata, kCallbacksRva - kTlsDirRva + 8 * i, callbacks[i]);

    // .data: IAT (filled by the binder) and the TLS index (0xFFFFFFFF until the loader writes 0).
    std::vector<uint8_t> data(0x200, 0);
    set32(data, kTlsIndexRva - kIatRva, 0xFFFFFFFFu);

    // .tls: template; the bytes after it in the section are garbage that must not be copied.
    std::vector<uint8_t> tls(0x100, 0xEE);
    for (size_t i = 0; i < kTlsTemplateSize; ++i) tls[i] = static_cast<uint8_t>(seed + i);

    std::vector<Sec> secs = {
        {"<headers>", 0, 0x400, kProtRead, kSectionFlagHeaders, headers},
        {".text", 0x1000, 0x400, kProtRead | kProtExecute, 0, text},
        {".rdata", 0x2000, 0x100, kProtRead, 0, rdata},
        {".data", 0x3000, 0x200, kProtRead | kProtWrite, 0, data},
        {".tls", 0x4000, 0x100, kProtRead | kProtWrite, 0, tls},
    };
    return write_container(secs, kFlagHeadersMapped, base);
}

// ---------------------------------------------------------------------------
// Guest-convention functions
// ---------------------------------------------------------------------------

VirtualMemoryArena* g_arena = nullptr;

enum class Scenario {
    ReturnCode,
    ExitViaCrt,
    RequestExit,
    Trap,
    Unimplemented,
    DivideError,
    UnknownTarget,
    StackImbalance,
    BadStack,
    Unresolved,
    SurveyLoop,
    SurveyReturn,
    CrtAbort,
    LastError,
};

Scenario g_scenario = Scenario::ReturnCode;
std::vector<std::string> g_events;

struct Observed {
    uint64_t teb_self = 0, stack_base = 0, stack_limit = 0, pid = 0, tid = 0;
    uint64_t tls_pointer = 0, peb = 0, peb_image_base = 0, peb_params = 0;
    uint64_t tls_block = 0;
    std::vector<uint8_t> tls_bytes;
    uint32_t tls_index = 0;
    uint64_t callback_args[2][3] = {};
    uint32_t mxcsr = 0;
    std::u16string command_line;
} g_seen;

uint64_t load64(uint64_t va) {
    uint64_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
uint32_t load32(uint64_t va) {
    uint32_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
void store64(uint64_t va, uint64_t v) { CHECK(g_arena->write(va, &v, sizeof(v))); }
void store32(uint64_t va, uint32_t v) { CHECK(g_arena->write(va, &v, sizeof(v))); }

uint64_t& reg(GuestState* s, uint32_t r) { return s->gpr[r]; }
void push(GuestState* s, uint64_t v) {
    reg(s, GuestReg::Rsp) -= 8;
    store64(reg(s, GuestReg::Rsp), v);
}
void ret(GuestState* s) { reg(s, GuestReg::Rsp) += 8; }
// `call [rip + iat slot]` from `return_address - 6`.
void call_iat(GuestState* s, uint32_t slot_rva, uint64_t return_address) {
    push(s, return_address);
    __playgta6_indirect_call(s, load64(kBase + slot_rva));
}

// Inline TLS access as MSVC emits it: mov rax, gs:[0x58]; mov ecx, [index]; mov rax, [rax+rcx*8].
uint64_t tls_block_of(GuestState* s) {
    const uint64_t vector = load64(s->gs_base + 0x58);
    const uint32_t index = load32(kBase + kTlsIndexRva);
    return load64(vector + 8ull * index);
}

void sub_leaf(GuestState* s) {
    __playgta6_trap(s, kLeaf + 4);
}

void sub_mid(GuestState* s) {
    push(s, 0x1234);  // a saved register that is not a return address
    push(s, kMidCallsLeaf);
    sub_leaf(s);
    ret(s);
    reg(s, GuestReg::Rsp) += 8;
    ret(s);
}

void sub_callback(GuestState* s, int which) {
    g_events.push_back(which == 0 ? "callback1" : "callback2");
    g_seen.callback_args[which][0] = reg(s, GuestReg::Rcx);
    g_seen.callback_args[which][1] = reg(s, GuestReg::Rdx);
    g_seen.callback_args[which][2] = reg(s, GuestReg::R8);
    // TLS must already be usable in callbacks.
    if (which == 0) {
        const uint64_t block = tls_block_of(s);
        g_events.push_back(load32(block) == 0xA3A2A1A0u ? "callback1-tls-ok" : "callback1-tls-bad");
    }
    ret(s);
}
void sub_callback1(GuestState* s) { sub_callback(s, 0); }
void sub_callback2(GuestState* s) { sub_callback(s, 1); }

void sub_worker(GuestState* s) {
    // Second-thread body: report its TLS block through rax.
    reg(s, GuestReg::Rax) = tls_block_of(s);
    ret(s);
}

void sub_entry(GuestState* s) {
    g_events.push_back("entry");
    switch (g_scenario) {
        case Scenario::ReturnCode: {
            g_seen.teb_self = load64(s->gs_base + 0x30);
            g_seen.stack_base = load64(s->gs_base + 0x08);
            g_seen.stack_limit = load64(s->gs_base + 0x10);
            g_seen.pid = load64(s->gs_base + 0x40);
            g_seen.tid = load64(s->gs_base + 0x48);
            g_seen.tls_pointer = load64(s->gs_base + 0x58);
            g_seen.peb = load64(s->gs_base + 0x60);
            g_seen.peb_image_base = load64(g_seen.peb + 0x10);
            g_seen.peb_params = load64(g_seen.peb + 0x20);
            g_seen.tls_index = load32(kBase + kTlsIndexRva);
            if (g_seen.tls_pointer != 0) {
                g_seen.tls_block = tls_block_of(s);
                g_seen.tls_bytes.resize(kTlsTemplateSize + kTlsZeroFill);
                CHECK(g_arena->read(g_seen.tls_block, g_seen.tls_bytes.data(), g_seen.tls_bytes.size()));
            }
            g_seen.mxcsr = s->mxcsr;
            // RTL_USER_PROCESS_PARAMETERS.CommandLine
            const uint16_t length = static_cast<uint16_t>(load32(g_seen.peb_params + 0x70));
            const uint64_t buffer = load64(g_seen.peb_params + 0x78);
            g_seen.command_line.resize(length / 2);
            CHECK(g_arena->read(buffer, g_seen.command_line.data(), length));
            reg(s, GuestReg::Rax) = 0xFFFFFFFF00000000ull | 42;  // exit code = eax
            ret(s);
            return;
        }
        case Scenario::ExitViaCrt:
            reg(s, GuestReg::Rcx) = 7;
            call_iat(s, kSlotExit, kEntryCallsIat);
            g_events.push_back("after-exit");  // must not run
            ret(s);
            return;
        case Scenario::RequestExit:
            pg6_guest_boot_request_exit(9);
            g_events.push_back("after-request-exit");
            ret(s);
            return;
        case Scenario::Trap:
            push(s, kEntryStaleValue);  // stale non-return address
            push(s, kEntryCallsMid);
            sub_mid(s);
            ret(s);
            return;
        case Scenario::Unimplemented:
            __playgta6_unimplemented(s, kEntry + 0x20);
        case Scenario::DivideError:
            __playgta6_divide_error(s, kEntry + 0x24);
        case Scenario::UnknownTarget:
            push(s, kEntryCallsIat);
            __playgta6_indirect_call(s, kUnknownInsideEntry);
            ret(s);
            return;
        case Scenario::StackImbalance:
            reg(s, GuestReg::Rax) = 1;
            return;  // forgets `ret`: rsp is 8 bytes too low
        case Scenario::BadStack:
            reg(s, GuestReg::Rsp) = 0x20;
            call_guest(*g_arena, *s, kMid);
            return;
        case Scenario::Unresolved:
            call_iat(s, kSlotMissingOne, kEntryCallsIat);
            ret(s);
            return;
        case Scenario::SurveyLoop:
            for (int i = 0; i < 1000; ++i) {
                call_iat(s, kSlotMissingTwo, kEntryCallsIat);
                call_iat(s, kSlotMissingOne, kEntryCallsIat);
                CHECK(reg(s, GuestReg::Rax) == 0);
            }
            ret(s);
            return;
        case Scenario::SurveyReturn:
            reg(s, GuestReg::Rax) = 0x55;
            call_iat(s, kSlotMissingTwo, kEntryCallsIat);
            CHECK(reg(s, GuestReg::Rax) == 0);
            call_iat(s, kSlotMissingOne, kEntryCallsIat);
            call_iat(s, kSlotMissingTwo, kEntryCallsIat);
            reg(s, GuestReg::Rax) = 5;
            ret(s);
            return;
        case Scenario::CrtAbort:
            call_iat(s, kSlotAbort, kEntryCallsIat);
            ret(s);
            return;
        case Scenario::LastError: {
            reg(s, GuestReg::Rcx) = 1234;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            const uint32_t from_teb = load32(s->gs_base + 0x68);  // inline gs:[0x68] read
            store32(s->gs_base + 0x68, 77);                        // inline write
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            reg(s, GuestReg::Rax) = (reg(s, GuestReg::Rax) == 77 && from_teb == 1234) ? 0 : 1;
            ret(s);
            return;
        }
    }
}

// Import thunks (R5 shape: called after the guest pushed its return address).
void thunk_exit(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    pg6_crt_exit(static_cast<int32_t>(reg(s, GuestReg::Rcx)));
    ret(s);
}
void thunk_abort(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    pg6_crt_abort();
    ret(s);
}
void thunk_set_last_error(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    os::win32::SetLastError(static_cast<uint32_t>(reg(s, GuestReg::Rcx)));
    ret(s);
}
void thunk_get_last_error(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    reg(s, GuestReg::Rax) = os::win32::GetLastError();
    ret(s);
}
void thunk_generated_unresolved(void*) {
    std::fprintf(stderr, "generated unresolved thunk must be replaced by the boot\n");
    std::abort();
}

const LiftedFunctionEntry kTable[] = {
    {kEntry, &sub_entry},         {kMid, &sub_mid},       {kLeaf, &sub_leaf},
    {kCallback1, &sub_callback1}, {kCallback2, &sub_callback2}, {kWorker, &sub_worker},
};

const BootImport kImports[] = {
    {kSlotExit, kPg6BootImportBound, 0, "api-ms-win-crt-runtime-l1-1-0.dll", "exit", "pg6_crt_exit", &thunk_exit},
    {kSlotSetLastError, kPg6BootImportBound, 0, "KERNEL32.dll", "SetLastError", "pg6_win32_set_last_error",
     &thunk_set_last_error},
    {kSlotGetLastError, kPg6BootImportBound, 0, "KERNEL32.dll", "GetLastError", "pg6_win32_get_last_error",
     &thunk_get_last_error},
    {kSlotMissingOne, 0, 0, "KERNEL32.dll", "MissingOne", nullptr, &thunk_generated_unresolved},
    {kSlotMissingTwo, kPg6BootImportDelayLoad, 0, "USER32.dll", "MissingTwo", nullptr,
     &thunk_generated_unresolved},
    {kSlotAbort, kPg6BootImportBound, 0, "api-ms-win-crt-runtime-l1-1-0.dll", "abort", "pg6_crt_abort",
     &thunk_abort},
};

std::vector<uint8_t> g_image;

BootConfig make_config(os::pgim::ImageReader& reader) {
    BootConfig cfg;
    BootModule exe;
    exe.image = &reader;
    exe.name = "boot_test.exe";
    exe.lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kTable));
    exe.imports = std::span<const BootImport>(kImports);
    cfg.modules.push_back(std::move(exe));
    cfg.arena = g_arena;
    cfg.command_line = "boot_test.exe -flag \xC3\xA4";  // non-ASCII survives as UTF-16
    cfg.working_directory = "C:\\game";
    cfg.log_progress = false;
    return cfg;
}

struct Run {
    BootResult result;
    Json report;
};

Run boot_scenario(Scenario scenario, void (*tweak)(BootConfig&) = nullptr) {
    g_scenario = scenario;
    g_events.clear();
    g_seen = Observed{};
    os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
    BootConfig cfg = make_config(reader);
    if (tweak != nullptr) {
        tweak(cfg);
    }
    Run run;
    run.result = run_boot(cfg);
    bool ok = false;
    run.report = parse_json(run.result.report_json, &ok);
    CHECK(ok);
    CHECK(run.report["schema"].text == "playgta6-boot-report");
    CHECK(run.report["schema_version"].number == 1);
    CHECK(run.report["status"].text == boot_status_name(run.result.status));
    CHECK(!run.result.summary.empty());
    if (!ok) {
        std::fprintf(stderr, "%s\n", run.result.report_json.c_str());
    }
    return run;
}

void check_fault(const Run& run, const char* kind, uint64_t va, uint64_t function_entry) {
    const Json& fault = run.report["fault"];
    CHECK(fault.type == Json::Object);
    CHECK(fault["kind"].text == kind);
    CHECK(fault["va"].hex() == va);
    if (function_entry != 0) {
        CHECK(fault["function"]["entry"].hex() == function_entry);
        CHECK(fault["function"]["offset"].hex() == va - function_entry);
    }
    const Json& regs = run.report["registers"];
    CHECK(regs.type == Json::Object);
    CHECK(regs.has("rax") && regs.has("r15") && regs.has("rsp") && regs.has("gs_base"));
    CHECK(regs["gs_base"].hex() == run.report["startup"]["main_thread"]["teb"].hex());
    if (std::string(kind) != "unresolved_import") {
        CHECK(run.result.status == BootStatus::Fault || run.result.status == BootStatus::GuestAbort);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_layout_offsets() {
    // The static_asserts in boot.h pin the layouts; spot-check the sizes here too.
    CHECK(sizeof(win64::Teb64) == 0x2000);
    CHECK(sizeof(win64::Peb64) == 0x1000);
    CHECK(sizeof(BootImport) == 48);
}

void test_clean_boot_and_teb() {
    const Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 42);
    CHECK(run.report["exit_code"].number == 42);
    CHECK(run.report["fault"].type == Json::Null);
    CHECK(run.report["phase"].text == "entry");
    {
        const Json& ms = run.report["milestones"];
        CHECK(ms.items.size() == 3 && ms[0]["milestone"].text == "PGIM_LOAD" &&
              ms[1]["milestone"].text == "TLS_INIT" && ms[2]["milestone"].text == "ENTRY_CALL");
    }
    // TLS callbacks ran first, in order, with (image base, DLL_PROCESS_ATTACH, null).
    CHECK(g_events.size() == 4);
    CHECK(g_events.size() == 4 && g_events[0] == "callback1" && g_events[1] == "callback1-tls-ok" &&
          g_events[2] == "callback2" && g_events[3] == "entry");
    for (int i = 0; i < 2; ++i) {
        CHECK(g_seen.callback_args[i][0] == kBase);
        CHECK(g_seen.callback_args[i][1] == win64::kDllProcessAttach);
        CHECK(g_seen.callback_args[i][2] == 0);
    }
    CHECK(run.result.stats.tls_callbacks == 2);
    CHECK(run.result.stats.tls_callbacks_run == 2);
    CHECK(run.result.stats.entry_started == 1);
    // TEB through gs.
    const Json& thread = run.report["startup"]["main_thread"];
    const uint64_t teb = thread["teb"].hex();
    CHECK(teb != 0 && teb % 0x1000 == 0);
    CHECK(g_seen.teb_self == teb);
    CHECK(g_seen.stack_base == thread["stack_base"].hex());
    CHECK(g_seen.stack_limit == thread["stack_limit"].hex());
    CHECK(g_seen.stack_base - g_seen.stack_limit == kPeStackReserve);  // PE SizeOfStackReserve
    CHECK(g_seen.pid == 0x1000);
    CHECK(g_seen.tid == os::win32::GetCurrentThreadId());
    CHECK(g_seen.peb == run.report["startup"]["process"]["peb"].hex());
    CHECK(g_seen.peb_image_base == kBase);
    CHECK(g_seen.mxcsr == kGuestMxcsrDefault);
    CHECK(g_seen.command_line == u"boot_test.exe -flag \u00E4");
    // Implicit TLS through gs:[0x58].
    CHECK(g_seen.tls_index == 0);
    CHECK(g_seen.tls_pointer == thread["tls_vector"].hex());
    CHECK(g_seen.tls_block == thread["tls_block"].hex());
    CHECK(g_seen.tls_block % 16 == 0);
    bool template_ok = g_seen.tls_bytes.size() == kTlsTemplateSize + kTlsZeroFill;
    for (size_t i = 0; template_ok && i < g_seen.tls_bytes.size(); ++i) {
        template_ok = g_seen.tls_bytes[i] == (i < kTlsTemplateSize ? template_byte(i) : 0);
    }
    CHECK(template_ok);
    const Json& tls = run.report["startup"]["modules"][0]["tls"];
    CHECK(tls["present"].boolean);
    CHECK(tls["template_size"].number == kTlsTemplateSize);
    CHECK(tls["zero_fill"].number == kTlsZeroFill);
    CHECK(tls["alignment"].number == 16);
    CHECK(tls["callbacks"].items.size() == 2 && tls["callbacks"][0].hex() == kCallback1);
    // Startup summary.
    const Json& imports = run.report["startup"]["imports"];
    CHECK(imports["total"].number == 6);
    CHECK(imports["bound"].number == 4);
    CHECK(imports["unresolved"].number == 2);
    CHECK(imports["by_dll"].items.size() == 3);
    CHECK(run.report["startup"]["functions_registered"].number == 6);
    CHECK(run.report["startup"]["image_base"].hex() == kBase);
    // The process keeps the guest thread's last-error slot bound until reset.
    CHECK(os::win32::GetLastError() == 0);
}

void test_exit_paths() {
    Run run = boot_scenario(Scenario::ExitViaCrt);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 7);
    CHECK(run.report["exit_code"].number == 7);
    CHECK(g_events.size() == 4 && g_events.back() == "entry");  // nothing after exit ran
    CHECK(run.report["counters"]["import_calls"].number == 1);
    CHECK(run.report["counters"]["recent"].items.size() == 1);
    CHECK(run.report["counters"]["recent"][0]["symbol"].text == "exit");

    run = boot_scenario(Scenario::RequestExit);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 9);
    CHECK(g_events.back() == "entry");

    run = boot_scenario(Scenario::CrtAbort);
    CHECK(run.result.status == BootStatus::GuestAbort);
    CHECK(run.report["fault"]["kind"].text == "crt_abort");
    CHECK(run.report["exit_code"].type == Json::Null);
}

void test_fault_reports() {
    // Trap two calls deep: function + backtrace.
    Run run = boot_scenario(Scenario::Trap);
    CHECK(run.result.status == BootStatus::Fault);
    CHECK(run.report["phase"].text == "entry");
    check_fault(run, "trap", kLeaf + 4, kLeaf);
    CHECK(run.report["fault"]["trap_kind"].text == "GUEST_TRAP");
    const Json& bt = run.report["backtrace"];
    CHECK(bt["method"].text == "stack-scan-call-verified");
    CHECK(bt["frames"].items.size() == 2);
    CHECK(bt["frames"][0]["return_address"].hex() == kMidCallsLeaf);
    CHECK(bt["frames"][0]["function"].hex() == kMid);
    CHECK(bt["frames"][0]["offset"].hex() == kMidCallsLeaf - kMid);
    CHECK(bt["frames"][1]["return_address"].hex() == kEntryCallsMid);
    CHECK(bt["frames"][1]["function"].hex() == kEntry);
    CHECK(bt["frames"][0]["stack_address"].hex() < bt["frames"][1]["stack_address"].hex());
    CHECK(run.result.summary.find("sub_140001180+0x4") != std::string::npos);

    run = boot_scenario(Scenario::Unimplemented);
    check_fault(run, "unimplemented", kEntry + 0x20, kEntry);
    run = boot_scenario(Scenario::DivideError);
    check_fault(run, "divide_error", kEntry + 0x24, kEntry);

    run = boot_scenario(Scenario::UnknownTarget);
    check_fault(run, "unknown_target", kUnknownInsideEntry, kEntry);
    CHECK(run.report["fault"]["trap_kind"].text == "INDIRECT_CALL_MISS");
    CHECK(run.report["fault"]["detail"].hex() == kEntry);  // nearest entry below the target
    CHECK(run.report["backtrace"]["frames"].items.size() == 1);
    CHECK(run.report["backtrace"]["frames"][0]["return_address"].hex() == kEntryCallsIat);

    run = boot_scenario(Scenario::StackImbalance);
    check_fault(run, "stack_imbalance", kEntry, kEntry);

    run = boot_scenario(Scenario::BadStack);
    check_fault(run, "bad_stack", kMid, kMid);
    CHECK(run.report["fault"]["detail"].hex() == 0x20);

    // Unresolved import: DLL!symbol, the call site and no survey entry.
    run = boot_scenario(Scenario::Unresolved);
    CHECK(run.result.status == BootStatus::Fault);
    check_fault(run, "unresolved_import", kEntryCallsIat, kEntry);
    const Json& imp = run.report["fault"]["import"];
    CHECK(imp["dll"].text == "KERNEL32.dll");
    CHECK(imp["symbol"].text == "MissingOne");
    CHECK(imp["module"].text == "boot_test.exe");
    CHECK(run.report["fault"]["trap_kind"].text == "UNRESOLVED_IMPORT");
    CHECK(imp["slot_rva"].hex() == kSlotMissingOne);
    CHECK(!imp["delay_load"].boolean);
    CHECK(run.report["counters"]["imports"].items.size() == 1);
    CHECK(run.report["counters"]["imports"][0]["resolved"].type == Json::Bool &&
          !run.report["counters"]["imports"][0]["resolved"].boolean);
    CHECK(run.report["survey"]["unresolved_called"].items.empty());
    CHECK(run.result.summary.find("KERNEL32.dll!MissingOne") != std::string::npos);

    // A fault in a TLS callback is attributed to that phase (entry never starts).
    run = boot_scenario(Scenario::ReturnCode, [](BootConfig& cfg) {
        static const LiftedFunctionEntry kBroken[] = {{kEntry, &sub_entry}, {kCallback1, &sub_leaf}};
        cfg.modules[0].lifted_tables.clear();
        cfg.modules[0].lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kBroken));
    });
    CHECK(run.result.status == BootStatus::Fault);
    CHECK(run.report["phase"].text == "tls_callback");
    CHECK(run.result.stats.entry_started == 0);
    CHECK(run.report["fault"]["kind"].text == "trap");
}

void test_survey_mode() {
    // Bounded: the loop would make 2000 import calls; the bound stops it.
    Run run = boot_scenario(Scenario::SurveyLoop, [](BootConfig& cfg) {
        cfg.survey = true;
        cfg.max_dispatches = 50;
        cfg.recent_import_count = 4;
    });
    CHECK(run.result.status == BootStatus::LimitReached);
    CHECK(run.report["fault"]["kind"].text == "dispatch_limit");
    CHECK(run.report["fault"]["trap_kind"].type == Json::Null);
    CHECK(run.report["survey"]["enabled"].boolean);
    const Json& called = run.report["survey"]["unresolved_called"];
    CHECK(called.items.size() == 2);
    CHECK(called[0]["symbol"].text == "MissingTwo" && called[0]["dll"].text == "USER32.dll");
    CHECK(called[1]["symbol"].text == "MissingOne");
    CHECK(called[0]["first_seq"].number == 1 && called[1]["first_seq"].number == 2);
    // Dispatches 1-2 are the TLS callbacks, 3 the entry point, 4.. the import calls; dispatch 51
    // (import call 48) hits the bound before its thunk runs.
    CHECK(called[0]["count"].number + called[1]["count"].number == 47);
    CHECK(run.report["counters"]["dispatches"].number == 51);
    CHECK(run.report["counters"]["import_calls"].number == 48);
    CHECK(run.report["counters"]["recent"].items.size() == 4);
    CHECK(run.report["counters"]["recent"][3]["seq"].number == 48);
    CHECK(run.report["counters"]["recent"][0]["seq"].number == 45);
    CHECK(run.result.stats.unresolved_called == 2);

    // Unbounded: survives past unresolved calls and returns normally.
    run = boot_scenario(Scenario::SurveyReturn, [](BootConfig& cfg) { cfg.survey = true; });
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 5);
    const Json& list = run.report["survey"]["unresolved_called"];
    CHECK(list.items.size() == 2);
    CHECK(list[0]["symbol"].text == "MissingTwo" && list[0]["count"].number == 2);
    CHECK(list[1]["symbol"].text == "MissingOne" && list[1]["count"].number == 1);

    // Wall-time bound alone (checked every 256 dispatches).
    run = boot_scenario(Scenario::SurveyLoop, [](BootConfig& cfg) {
        cfg.survey = true;
        cfg.max_wall_ms = 1;
    });
    CHECK(run.result.status == BootStatus::LimitReached || run.result.status == BootStatus::Exited);
}

void test_last_error_shared_with_teb() {
    const Run run = boot_scenario(Scenario::LastError);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 0);
}

void test_second_thread() {
    Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    const uint64_t main_teb = run.report["startup"]["main_thread"]["teb"].hex();
    const uint64_t main_block = run.report["startup"]["main_thread"]["tls_block"].hex();
    // The main thread changes its TLS copy; the new thread must still get the pristine template.
    store32(main_block, 0x11111111u);

    GuestThreadContext worker{};
    CHECK(prepare_guest_thread(0x10000, 77, worker));
    CHECK(worker.teb != 0 && worker.teb != main_teb);
    CHECK(worker.tls_block != 0 && worker.tls_block != main_block);
    CHECK(worker.thread.state.gs_base == worker.teb);
    CHECK(load64(worker.teb + 0x30) == worker.teb);
    CHECK(load64(worker.teb + 0x48) == 77);
    CHECK(load64(worker.teb + 0x58) == worker.tls_vector);
    CHECK(load64(worker.teb + 0x60) == run.report["startup"]["process"]["peb"].hex());
    CHECK(load64(worker.teb + 0x08) == worker.thread.stack_top);
    CHECK(load32(worker.tls_block) == 0xA3A2A1A0u);
    CHECK(load32(worker.tls_block + kTlsTemplateSize) == 0);
    // Guest code on the new thread reaches its own block through gs.
    CHECK(call_guest(*g_arena, worker.thread.state, kWorker) == worker.tls_block);
    // Its last-error slot is separate from the main thread's.
    GuestThreadContext* previous = activate_guest_thread(&worker);
    os::win32::SetLastError(5);
    CHECK(load32(worker.teb + 0x68) == 5);
    CHECK(load32(main_teb + 0x68) != 5);
    activate_guest_thread(previous);
    CHECK(run_tls_callbacks(worker, win64::kDllThreadAttach) == 2);
    CHECK(g_seen.callback_args[0][1] == win64::kDllThreadAttach);
    release_guest_thread(worker);
    CHECK(worker.teb == 0);

    // Host threads that O16 gives a guest thread lazily (CreateThread, pool callbacks) get a TEB
    // and TLS from the boot, and the TLS callbacks run with DLL_THREAD_ATTACH.
    g_events.clear();
    g_seen.callback_args[0][1] = 0;
    uint64_t lazy_teb = 0, lazy_self = 0, lazy_block = 0, lazy_rax = 0;
    uint32_t lazy_first = 0;
    std::thread host([&] {
        GuestState& state = com::current_guest_state();
        lazy_teb = state.gs_base;
        lazy_self = load64(state.gs_base + 0x30);
        lazy_block = load64(load64(state.gs_base + 0x58));
        lazy_first = load32(lazy_block);
        lazy_rax = call_guest(*g_arena, state, kWorker);
    });
    host.join();
    CHECK(lazy_teb != 0 && lazy_teb != main_teb);
    CHECK(lazy_self == lazy_teb);
    CHECK(lazy_block != 0 && lazy_block != main_block);
    CHECK(lazy_first == 0xA3A2A1A0u);
    CHECK(lazy_rax == lazy_block);
    CHECK(g_events.size() == 3 && g_events[0] == "callback1");
    CHECK(g_seen.callback_args[0][1] == win64::kDllThreadAttach);
}

void test_no_tls_and_load_failure() {
    const std::vector<uint8_t> saved = g_image;
    g_image = build_image(false);
    Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(!run.report["startup"]["modules"][0]["tls"]["present"].boolean);
    CHECK(run.report["startup"]["main_thread"]["tls_vector"].type == Json::Null);
    CHECK(g_seen.tls_pointer == 0);
    CHECK(g_events.size() == 1 && g_events[0] == "entry");

    g_image.assign(g_image.size(), 0);  // not a PGIM container
    run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::LoadFailed);
    CHECK(run.report["phase"].text == "load");
    CHECK(run.report["startup"]["image_base"].type == Json::Null);
    g_image = saved;

    // C ABI: in-memory image, one table, quiet.
    g_scenario = Scenario::ReturnCode;
    const Pg6LiftedFunctionEntry* tables[] = {kTable};
    const size_t counts[] = {std::size(kTable)};
    Pg6GuestBootModule module{};
    module.image_bytes = g_image.data();
    module.image_size = g_image.size();
    module.name = "boot_test.exe";
    module.lifted_tables = tables;
    module.lifted_table_counts = counts;
    module.lifted_table_count = 1;
    module.imports = kImports;
    module.import_count = std::size(kImports);
    Pg6GuestBootConfig cfg{};
    cfg.modules = &module;
    cfg.module_count = 1;
    cfg.quiet = 1;
    Pg6GuestBootStats stats{};
    os::win32::set_vmem_arena(g_arena);
    CHECK(pg6_guest_boot_run(&cfg, &stats) == PG6_BOOT_EXITED);
    CHECK(stats.exit_code == 42);
    CHECK(stats.imports_total == 6);
    char small[16];
    const size_t full = pg6_guest_boot_report_json(small, sizeof(small));
    CHECK(full > sizeof(small) && std::strlen(small) == sizeof(small) - 1);
}

// A second guest module (a DLL) with its own TLS directory, callback and IAT.
constexpr uint64_t kDllBase = 0x180000000ULL;
constexpr uint64_t kDllCallback = kDllBase + 0x1000;

uint32_t g_dll_tls_index = 0;
uint32_t g_dll_tls_first = 0;

void sub_dll_callback(GuestState* s) {
    g_events.push_back("dll-callback");
    g_dll_tls_index = load32(kDllBase + kTlsIndexRva);
    const uint64_t vector = load64(s->gs_base + 0x58);
    g_dll_tls_first = load32(load64(vector + 8ull * g_dll_tls_index));
    ret(s);
}

const LiftedFunctionEntry kDllTable[] = {{kDllCallback, &sub_dll_callback}};
const BootImport kDllImports[] = {
    {kIatRva, 0, 0, "VCRUNTIME140.dll", "memcpy", nullptr, &thunk_generated_unresolved},
};

void test_two_modules() {
    const std::vector<uint8_t> dll_image = build_image(true, kDllBase, {kDllCallback}, 0xB0);
    os::pgim::MemoryImageReader dll_reader(dll_image.data(), dll_image.size());
    g_scenario = Scenario::ReturnCode;
    g_events.clear();
    g_seen = Observed{};
    os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
    BootConfig cfg = make_config(reader);
    BootModule dll;
    dll.image = &dll_reader;
    dll.name = "helper.dll";
    dll.lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kDllTable));
    dll.imports = std::span<const BootImport>(kDllImports);
    cfg.modules.push_back(std::move(dll));
    const BootResult result = run_boot(cfg);
    bool ok = false;
    const Json report = parse_json(result.report_json, &ok);
    CHECK(ok);
    CHECK(result.status == BootStatus::Exited);
    CHECK(result.stats.exit_code == 42);
    // DLL callbacks first, then the executable's, then the entry point.
    CHECK(g_events.size() == 5 && g_events[0] == "dll-callback" && g_events[1] == "callback1" &&
          g_events[4] == "entry");
    // TLS indices in module order: the executable 0, the DLL 1; each block has its own template.
    CHECK(g_seen.tls_index == 0);
    CHECK(g_dll_tls_index == 1);
    CHECK(g_dll_tls_first == 0xB3B2B1B0u);
    CHECK(g_seen.tls_bytes.size() > 0 && g_seen.tls_bytes[0] == 0xA0);
    const Json& modules = report["startup"]["modules"];
    CHECK(modules.items.size() == 2);
    CHECK(modules[1]["name"].text == "helper.dll");
    CHECK(modules[1]["image_base"].hex() == kDllBase);
    CHECK(modules[1]["tls"]["index"].number == 1);
    CHECK(modules[1]["tls"]["callbacks_run"].number == 1);
    CHECK(modules[1]["imports"]["unresolved"].number == 1);
    CHECK(modules[0]["tls"]["index"].number == 0);
    CHECK(report["startup"]["functions_registered"].number == 7);
    CHECK(report["startup"]["imports"]["total"].number == 7);
    // The DLL's IAT slot got its own synthetic address (bound to the boot's unresolved thunk).
    SyntheticTarget target;
    CHECK(describe_synthetic(load64(kDllBase + kIatRva), &target));
    CHECK(target.is_import && target.image_base == kDllBase && target.slot_rva == kIatRva);
    CHECK(result.stats.tls_callbacks == 3);
}

void test_host_fault_mapping() {
    Run run = boot_scenario(Scenario::ReturnCode);
    const auto host_mid = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&sub_mid));
    uint64_t offset = 99;
    // Function addresses of hand-written functions are not ordered like guest VAs, but an exact
    // function address always maps back to its own entry.
    CHECK(guest_function_for_host_pc(host_mid, &offset) == kMid);
    CHECK(offset == 0);
    CHECK(report_host_fault("host_access_violation", 0xDEAD0000, host_mid));
    bool ok = false;
    const Json report = parse_json(last_report_json(), &ok);
    CHECK(ok);
    CHECK(report["status"].text == "fault");
    CHECK(report["fault"]["kind"].text == "host_access_violation");
    CHECK(report["fault"]["host_address"].hex() == 0xDEAD0000);
    CHECK(report["fault"]["function"]["entry"].hex() == kMid);
}

}  // namespace

int main() {
#ifdef __EMSCRIPTEN__
    // The default wasm arena is 2 GiB ([4, 6) GiB) but the helper DLL is mapped at 6 GiB.
    VirtualMemoryArena arena(os::kDefaultArenaBase, uint64_t{4} << 30);
#else
    VirtualMemoryArena arena;
#endif
    g_arena = &arena;
    os::win32::set_vmem_arena(&arena);
    g_image = build_image();

    test_layout_offsets();
    test_clean_boot_and_teb();
    test_exit_paths();
    test_fault_reports();
    test_survey_mode();
    test_last_error_shared_with_teb();
    test_second_thread();
    test_no_tls_and_load_failure();
    test_two_modules();
    test_host_fault_mapping();

    reset_boot();
    os::win32::set_vmem_arena(nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_guest_boot: all checks passed\n");
    return 0;
}
