#include "os/host_marshallers.h"

#include <cstdio>
#include <string>

#include "guest/com_bridge.h"
#include "os/crt_format.h"
#include "os/os_shim.h"

namespace gcom = playgta6::guest::com;
namespace cf = playgta6::os::crt_format;

namespace {

using gcom::CallContext;

// inet_ntoa: the shim's formatting is host-only (a host buffer), so format here into the calling
// thread's guest buffer. 16 bytes hold the longest "255.255.255.255" plus the terminator.
uint64_t inet_ntoa_impl(CallContext&, uint32_t s_addr) {
    thread_local uint64_t t_buffer = 0;
    char text[16];
    std::snprintf(text, sizeof(text), "%u.%u.%u.%u", s_addr & 0xFFu, (s_addr >> 8) & 0xFFu,
                  (s_addr >> 16) & 0xFFu, (s_addr >> 24) & 0xFFu);
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (t_buffer == 0) {
            t_buffer = gcom::guest_alloc(sizeof(text));
            if (t_buffer == 0) {
                return 0;  // out of guest memory: NULL, like a failed call
            }
        }
        if (gcom::write_guest(t_buffer, text, sizeof(text))) {
            return t_buffer;
        }
        t_buffer = 0;  // the guest memory behind the cached block is gone: allocate afresh
    }
    return 0;
}

// Variadic slot i of the call: r9, then the stack slots after the shadow space. Read with the
// checked accessor so a bad guest stack ends the formatting instead of faulting the guest.
class VarargSource final : public cf::ArgSource {
public:
    explicit VarargSource(const CallContext& ctx) : ctx_(ctx), strings_(0) {}

    bool next(uint64_t& slot) override {
        if (index_ >= kHcTraceMaxArgs) {
            return false;
        }
        const unsigned i = index_++;
        if (i == 0) {
            slot = ctx_.int_arg(3);
            return true;
        }
        const uint64_t va = ctx_.entry_rsp + playgta6::kGuestStackArgOffset + 8ull * i;
        return gcom::read_guest(va, &slot, sizeof(slot));
    }
    bool narrow(uint64_t address, size_t max, std::string& out) override {
        return strings_.narrow(address, max, out);
    }
    bool wide(uint64_t address, size_t max, std::u16string& out) override {
        return strings_.wide(address, max, out);
    }

private:
    const CallContext& ctx_;
    cf::GuestArgSource strings_;  // only for its string readers (the slots come from the call)
    unsigned index_ = 0;
};

const char* level_name(uint32_t level) {
    switch (level) {
        case 0: return "Off";
        case 1: return "Error";
        case 2: return "Warning";
        case 3: return "Important";
        case 4: return "Information";
        case 5: return "Verbose";
        default: return "Level?";
    }
}

// Printable ASCII only, so a hostile guest string cannot forge log lines or terminal escapes.
std::string sanitize(const std::string& in, size_t limit) {
    std::string out;
    for (const char c : in) {
        if (out.size() >= limit) {
            out += "...";
            break;
        }
        const auto u = static_cast<unsigned char>(c);
        out += (u >= 0x20 && u < 0x7F) ? c : '?';
    }
    return out;
}

void hc_trace_impl(CallContext& ctx, uint64_t area, uint32_t level, uint64_t format) {
    cf::GuestArgSource reader(0);
    // HCTraceImplArea { const char* Name; HCTraceLevel Verbosity; }
    std::string area_name = "?";
    uint64_t name_va = 0;
    if (area != 0 && gcom::read_guest(area, &name_va, sizeof(name_va)) && name_va != 0) {
        std::string name;
        if (reader.narrow(name_va, 64, name)) {
            area_name = sanitize(name, 64);
        }
    }

    std::string raw;
    std::string message;
    const char* note = "";
    if (format == 0 || !reader.narrow(format, kHcTraceMaxFormat, raw)) {
        message = "<unreadable format>";
    } else {
        VarargSource args(ctx);
        const cf::Status status = cf::format_narrow(raw, 0, args, message);
        if (status != cf::Status::Ok) {
            message = raw;
            note = " (format error, arguments not applied)";
        }
    }
    // The trace line ends with a newline in libHttpClient's own formatting; drop it.
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    playgta6::os::log("libHttpClient [" + area_name + "] " + level_name(level) + ": " +
                      sanitize(message, kHcTraceMaxMessage) + note);
}

}  // namespace

extern "C" {

void pg6_gthunk_inet_ntoa(void* state) {
    gcom::function_thunk<&inet_ntoa_impl>()(state);
}

void pg6_gthunk_hc_trace_impl_message(void* state) {
    gcom::function_thunk<&hc_trace_impl>()(state);
}

}  // extern "C"
