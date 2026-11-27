#include "guest/guest_probe.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

namespace playgta6::guest {

namespace {

constexpr uint64_t kMinValidVa = 0x10000;            // the null page and below are never guest memory
constexpr uint64_t kMaxValidVa = 0x00007fffffffffffull;  // canonical user half
constexpr uint64_t kPage = 4096;

bool parse_hex(std::string_view text, uint64_t* out) {
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2);
    }
    if (text.empty() || text.size() > 16) {
        return false;
    }
    uint64_t v = 0;
    for (char c : text) {
        unsigned d = 0;
        if (c >= '0' && c <= '9') {
            d = static_cast<unsigned>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            d = static_cast<unsigned>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            d = static_cast<unsigned>(c - 'A' + 10);
        } else {
            return false;
        }
        v = (v << 4) | d;
    }
    *out = v;
    return true;
}

bool parse_dec(std::string_view text, uint64_t* out) {
    if (text.empty() || text.size() > 18) {
        return false;
    }
    uint64_t v = 0;
    for (char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    *out = v;
    return true;
}

bool name_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
           c == '-';
}

bool set_error(std::string* error, const std::string& msg) {
    if (error != nullptr) {
        *error = msg;
    }
    return false;
}

std::string hex_u64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "\"0x%" PRIx64 "\"", v);
    return b;
}

std::string hex_bytes(const uint8_t* p, size_t n) {
    static const char kDigits[] = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(kDigits[p[i] >> 4]);
        s.push_back(kDigits[p[i] & 15]);
    }
    return s;
}

}  // namespace

const char* read_status_name(ReadStatus s) {
    switch (s) {
        case ReadStatus::Ok: return "ok";
        case ReadStatus::Partial: return "partial";
        case ReadStatus::Unmapped: return "unmapped";
        case ReadStatus::NullPage: return "null_page";
        case ReadStatus::OutOfRange: return "out_of_range";
        case ReadStatus::Overflow: return "overflow";
    }
    return "unmapped";
}

SafeReadResult safe_read(const ProbeReadFn& read, uint64_t addr, void* dst, size_t len) {
    SafeReadResult r;
    if (len == 0) {
        r.status = ReadStatus::Ok;
        return r;
    }
    if (!read || dst == nullptr) {
        return r;
    }
    if (addr < kMinValidVa) {
        r.status = ReadStatus::NullPage;
        return r;
    }
    if (addr > kMaxValidVa) {
        r.status = ReadStatus::OutOfRange;
        return r;
    }
    if (len - 1 > kMaxValidVa - addr) {
        r.status = ReadStatus::Overflow;
        return r;
    }
    if (read(addr, dst, len)) {
        r.status = ReadStatus::Ok;
        r.bytes_ok = len;
        return r;
    }
    // Find the readable prefix, a page at a time (the first chunk ends at the page boundary).
    auto* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < len) {
        const uint64_t at = addr + done;
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(len - done, kPage - (at % kPage)));
        if (!read(at, out + done, chunk)) {
            break;
        }
        done += chunk;
    }
    r.bytes_ok = done;
    r.status = done == len ? ReadStatus::Ok : done > 0 ? ReadStatus::Partial : ReadStatus::Unmapped;
    return r;
}

int probe_reg_index(std::string_view name) {
    static const char* const kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                           "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    for (int i = 0; i < 16; ++i) {
        if (name == kNames[i]) {
            return i;
        }
    }
    return -1;
}

bool parse_probe_spec(std::string_view text, ProbeSpec* out, std::string* error) {
    ProbeSpec s;
    size_t pos = 0;
    bool first = true;
    bool named = false;
    while (pos <= text.size()) {
        size_t colon = text.find(':', pos);
        if (colon == std::string_view::npos) {
            colon = text.size();
        }
        const std::string_view f = text.substr(pos, colon - pos);
        pos = colon + 1;
        if (first) {
            first = false;
            const size_t plus = f.find('+');
            std::string_view addr = f;
            if (plus != std::string_view::npos && !(f.size() > 1 && f[0] == '0' && (f[1] == 'x' || f[1] == 'X'))) {
                const std::string_view mod = f.substr(0, plus);
                if (mod.empty() || !std::all_of(mod.begin(), mod.end(), name_char)) {
                    return set_error(error, "bad module name in address");
                }
                s.module = std::string(mod);
                addr = f.substr(plus + 1);
            }
            if (!parse_hex(addr, &s.addr) || s.addr == 0) {
                return set_error(error, "address must be a non-zero hex value");
            }
            continue;
        }
        if (f.empty()) {
            return set_error(error, "empty field");
        }
        if (f == "ret") {
            s.ret = true;
        } else if (f.rfind("args=", 0) == 0) {
            uint64_t n = 0;
            if (!parse_dec(f.substr(5), &n) || n > kProbeMaxArgs) {
                return set_error(error, "args=N needs N in 0..8");
            }
            s.args = static_cast<uint32_t>(n);
        } else if (f.rfind("max=", 0) == 0) {
            uint64_t n = 0;
            if (!parse_dec(f.substr(4), &n) || n == 0) {
                return set_error(error, "max=N needs a positive N");
            }
            s.max_lines = n;
        } else if (f.rfind("mem=", 0) == 0) {
            std::string_view m = f.substr(4);
            const size_t comma = m.find(',');
            if (comma == std::string_view::npos) {
                return set_error(error, "mem=REG+OFF,LEN needs a length");
            }
            std::string_view reg = m.substr(0, comma);
            uint64_t len = 0;
            if (!parse_dec(m.substr(comma + 1), &len) || len == 0 || len > kProbeMaxMemLen) {
                return set_error(error, "mem length must be 1..256");
            }
            ProbeMem pm;
            pm.len = static_cast<uint32_t>(len);
            const size_t sign = reg.find_first_of("+-");
            std::string_view regname = reg.substr(0, sign);
            if (sign != std::string_view::npos) {
                uint64_t off = 0;
                if (!parse_hex(reg.substr(sign + 1), &off) || off > 0x7fffffffull) {
                    return set_error(error, "mem offset must be hex up to 0x7fffffff");
                }
                pm.off = reg[sign] == '-' ? -static_cast<int64_t>(off) : static_cast<int64_t>(off);
            }
            const int idx = probe_reg_index(regname);
            if (idx < 0) {
                return set_error(error, "unknown register in mem=");
            }
            pm.reg = static_cast<uint32_t>(idx);
            if (s.mem.size() >= kProbeMaxMem) {
                return set_error(error, "at most 4 mem= reads per probe");
            }
            s.mem.push_back(pm);
        } else if (f.rfind("name=", 0) == 0 || (!named && std::all_of(f.begin(), f.end(), name_char))) {
            const std::string_view n = f.rfind("name=", 0) == 0 ? f.substr(5) : f;
            if (n.empty() || n.size() > 48 || !std::all_of(n.begin(), n.end(), name_char)) {
                return set_error(error, "bad probe name");
            }
            s.name = std::string(n);
            named = true;
        } else {
            return set_error(error, "unknown field '" + std::string(f) + "'");
        }
    }
    if (first || s.addr == 0) {
        return set_error(error, "missing address");
    }
    *out = std::move(s);
    return true;
}

std::string format_probe_line(const ProbeSpec& spec, uint64_t addr, const ProbeHit& hit, const ProbeReadFn& read) {
    char head[192];
    std::snprintf(head, sizeof(head), "{\"seq\":%" PRIu64 ",\"t_ms\":%" PRIu64 ",\"thread\":%u,\"src\":\"%s\",\"name\":\"",
                  hit.seq, hit.t_ms, hit.thread, hit.source);
    std::string o = head;
    o += spec.name.empty() ? "probe" : spec.name;
    o += "\",\"addr\":" + hex_u64(addr);

    const uint64_t rsp = hit.regs[4];
    uint64_t caller = 0;
    const SafeReadResult cr = safe_read(read, rsp, &caller, sizeof(caller));
    o += ",\"caller\":" + (cr.status == ReadStatus::Ok ? hex_u64(caller) : std::string("null"));

    if (spec.args != 0) {
        o += ",\"args\":[";
        static const uint32_t kArgRegs[4] = {1, 2, 8, 9};  // rcx rdx r8 r9
        for (uint32_t i = 0; i < spec.args; ++i) {
            if (i != 0) {
                o += ",";
            }
            if (i < 4) {
                o += hex_u64(hit.regs[kArgRegs[i]]);
            } else {
                // Return address at [rsp], then 0x20 bytes of shadow space, then the fifth argument.
                uint64_t v = 0;
                const SafeReadResult ar = safe_read(read, rsp + 0x28 + 8ull * (i - 4), &v, sizeof(v));
                o += ar.status == ReadStatus::Ok ? hex_u64(v) : std::string("null");
            }
        }
        o += "]";
    }
    if (spec.ret) {
        o += ",\"ret\":\"pending\"";
    }
    if (!spec.mem.empty()) {
        static const char* const kRegNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                                  "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
        o += ",\"mem\":[";
        for (size_t k = 0; k < spec.mem.size(); ++k) {
            const ProbeMem& m = spec.mem[k];
            const uint64_t at = hit.regs[m.reg] + static_cast<uint64_t>(m.off);
            uint8_t buf[kProbeMaxMemLen];
            const SafeReadResult r = safe_read(read, at, buf, m.len);
            char item[160];
            std::snprintf(item, sizeof(item), "%s{\"reg\":\"%s\",\"off\":%" PRId64 ",\"addr\":\"0x%" PRIx64 "\",\"len\":%u,\"status\":\"%s\"",
                          k == 0 ? "" : ",", kRegNames[m.reg], m.off, at, m.len, read_status_name(r.status));
            o += item;
            if (r.status == ReadStatus::Partial) {
                o += ",\"ok_bytes\":" + std::to_string(r.bytes_ok);
            }
            if (r.bytes_ok != 0) {
                o += ",\"hex\":\"" + hex_bytes(buf, r.bytes_ok) + "\"";
            }
            o += "}";
        }
        o += "]";
    }
    o += "}";
    return o;
}

std::string format_probe_exit_line(const ProbeSpec& spec, uint64_t addr, uint64_t seq, uint64_t t_ms, uint32_t thread,
                                   uint64_t entry_seq, bool paired, uint64_t rax, const uint64_t xmm0[2]) {
    char head[192];
    std::snprintf(head, sizeof(head), "{\"seq\":%" PRIu64 ",\"t_ms\":%" PRIu64 ",\"thread\":%u,\"src\":\"exit\",\"name\":\"",
                  seq, t_ms, thread);
    std::string o = head;
    o += spec.name.empty() ? "probe" : spec.name;
    o += "\",\"addr\":" + hex_u64(addr);
    o += ",\"entry_seq\":" + (paired && entry_seq != 0 ? std::to_string(entry_seq) : std::string("null"));
    o += std::string(",\"paired\":") + (paired ? "true" : "false");
    o += ",\"rax\":" + hex_u64(rax) + ",\"xmm0\":[" + hex_u64(xmm0[0]) + "," + hex_u64(xmm0[1]) + "]}";
    return o;
}

void ProbeSet::clear() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
    file_failed_ = false;
    specs_.clear();
    calls_.reset();
    entry_calls_.reset();
    logged_.reset();
    exit_calls_.reset();
    exit_unpaired_.reset();
    count_ = 0;
    min_ = max_ = 0;
    lines_.store(0, std::memory_order_relaxed);
    seq_.store(0, std::memory_order_relaxed);
    out_path_.clear();
}

void ProbeSet::configure(std::vector<ProbeSpec> specs, const std::string& out_path, uint64_t line_budget) {
    clear();
    if (specs.size() > kMaxProbes) {
        specs.resize(kMaxProbes);
    }
    std::stable_sort(specs.begin(), specs.end(),
                     [](const ProbeSpec& a, const ProbeSpec& b) { return a.addr < b.addr; });
    specs_ = std::move(specs);
    count_ = specs_.size();
    budget_ = line_budget;
    out_path_ = out_path;
    if (count_ == 0) {
        return;
    }
    min_ = specs_.front().addr;
    max_ = specs_.back().addr;
    calls_ = std::make_unique<std::atomic<uint64_t>[]>(count_);
    entry_calls_ = std::make_unique<std::atomic<uint64_t>[]>(count_);
    logged_ = std::make_unique<std::atomic<uint64_t>[]>(count_);
    exit_calls_ = std::make_unique<std::atomic<uint64_t>[]>(count_);
    exit_unpaired_ = std::make_unique<std::atomic<uint64_t>[]>(count_);
    for (size_t i = 0; i < count_; ++i) {
        calls_[i].store(0, std::memory_order_relaxed);
        entry_calls_[i].store(0, std::memory_order_relaxed);
        logged_[i].store(0, std::memory_order_relaxed);
        exit_calls_[i].store(0, std::memory_order_relaxed);
        exit_unpaired_[i].store(0, std::memory_order_relaxed);
    }
    if (!out_path_.empty()) {
        file_ = std::fopen(out_path_.c_str(), "wb");
        file_failed_ = file_ == nullptr;
    }
}

long ProbeSet::find(uint64_t addr) const noexcept {
    if (count_ == 0 || addr < min_ || addr > max_) {
        return -1;
    }
    size_t lo = 0, hi = count_;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (specs_[mid].addr < addr) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < count_ && specs_[lo].addr == addr ? static_cast<long>(lo) : -1;
}

bool ProbeSet::count_hit(size_t i, bool entry) noexcept {
    if (i >= count_) {
        return false;
    }
    const uint64_t n = calls_[i].fetch_add(1, std::memory_order_relaxed) + 1;
    if (entry) {
        entry_calls_[i].fetch_add(1, std::memory_order_relaxed);
    }
    if (n > specs_[i].max_lines) {
        return false;
    }
    return lines_.fetch_add(1, std::memory_order_relaxed) < budget_;
}

uint64_t ProbeSet::log_hit(size_t i, ProbeHit hit, const ProbeReadFn& read) {
    if (i >= count_) {
        return 0;
    }
    hit.seq = next_seq();
    const std::string line = format_probe_line(specs_[i], specs_[i].addr, hit, read);
    logged_[i].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_ != nullptr) {
        std::fwrite(line.data(), 1, line.size(), file_);
        std::fputc('\n', file_);
        std::fflush(file_);  // a fault right after must not lose the last hits
    }
    return hit.seq;
}

void ProbeSet::log_exit(size_t i, uint32_t thread, uint64_t t_ms, uint64_t entry_seq, bool paired, uint64_t rax,
                        const uint64_t xmm0[2]) noexcept {
    if (i >= count_) {
        return;
    }
    exit_calls_[i].fetch_add(1, std::memory_order_relaxed);
    if (paired) {
        lines_.fetch_add(1, std::memory_order_relaxed);  // bounded by the logged entries
    } else {
        const uint64_t n = exit_unpaired_[i].fetch_add(1, std::memory_order_relaxed) + 1;
        if (n > specs_[i].max_lines || lines_.fetch_add(1, std::memory_order_relaxed) >= budget_) {
            return;
        }
    }
    const std::string line = format_probe_exit_line(specs_[i], specs_[i].addr, next_seq(), t_ms, thread, entry_seq, paired,
                                                    rax, xmm0);
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_ != nullptr) {
        std::fwrite(line.data(), 1, line.size(), file_);
        std::fputc('\n', file_);
        std::fflush(file_);
    }
}

std::string ProbeSet::to_json() const {
    std::string o = "{\"enabled\": ";
    o += count_ != 0 ? "true" : "false";
    o += ", \"out\": \"";
    for (char c : out_path_) {
        if (c == '\\' || c == '"') {
            o.push_back('\\');
        }
        o.push_back(c == '\n' ? ' ' : c);
    }
    o += "\", \"line_budget\": " + std::to_string(budget_) + ", \"lines\": " +
         std::to_string(std::min(lines_.load(std::memory_order_relaxed), budget_)) + ", \"probes\": [";
    for (size_t i = 0; i < count_; ++i) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "%s{\"name\": \"%s\", \"addr\": \"0x%" PRIx64 "\", \"calls\": %" PRIu64 ", \"entry_calls\": %" PRIu64
                      ", \"exit_calls\": %" PRIu64 ", \"exit_unpaired\": %" PRIu64 ", \"logged\": %" PRIu64 ", \"rate_limited\": %" PRIu64 ", \"max_lines\": %" PRIu64 "}",
                      i == 0 ? "" : ", ", specs_[i].name.empty() ? "probe" : specs_[i].name.c_str(), specs_[i].addr,
                      calls(i), entry_calls(i), exit_calls(i), exit_unpaired(i), logged(i), calls(i) > logged(i) ? calls(i) - logged(i) : 0,
                      specs_[i].max_lines);
        o += buf;
    }
    return o + "]}";
}

}  // namespace playgta6::guest
