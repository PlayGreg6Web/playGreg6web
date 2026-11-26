#include "guest/export_counters.h"

#include <algorithm>
#include <cstring>

namespace playgta6::guest {

bool enumerate_pe_exports(const ExportReadFn& read, uint64_t base, std::vector<PeExport>& out) {
    out.clear();
    uint32_t lfanew = 0;
    uint8_t nt[0x90];
    if (!read(base + 0x3C, &lfanew, sizeof(lfanew)) || lfanew == 0 || lfanew > 0x1000 ||
        !read(base + lfanew, nt, sizeof(nt))) {
        return false;
    }
    uint32_t sig = 0;
    uint16_t magic = 0;
    std::memcpy(&sig, nt, 4);
    std::memcpy(&magic, nt + 0x18, 2);
    if (sig != 0x4550 || magic != 0x20B) {
        return false;
    }
    uint32_t dir_rva = 0;
    uint32_t dir_size = 0;
    std::memcpy(&dir_rva, nt + 0x18 + 0x70, 4);
    std::memcpy(&dir_size, nt + 0x18 + 0x74, 4);
    uint32_t ed[10];
    if (dir_rva == 0 || dir_size == 0 || !read(base + dir_rva, ed, sizeof(ed))) {
        return false;
    }
    const uint32_t n_funcs = ed[5];
    const uint32_t n_names = ed[6];
    if (n_names > 65536 || n_funcs > 65536) {
        return false;
    }
    for (uint32_t i = 0; i < n_names; ++i) {
        uint32_t name_rva = 0;
        uint16_t ord = 0;
        uint32_t func_rva = 0;
        if (!read(base + ed[8] + 4ULL * i, &name_rva, 4) || !read(base + ed[9] + 2ULL * i, &ord, 2) ||
            ord >= n_funcs || !read(base + ed[7] + 4ULL * ord, &func_rva, 4)) {
            continue;
        }
        if (func_rva == 0 || (func_rva >= dir_rva && func_rva < dir_rva + dir_size)) {
            continue;  // empty or a forwarder string
        }
        char name[128] = {};
        for (size_t k = 0; k + 1 < sizeof(name); ++k) {  // a name may end at the page edge
            if (!read(base + name_rva + k, &name[k], 1) || name[k] == '\0') {
                break;
            }
        }
        if (name[0] == '\0') {
            continue;
        }
        out.push_back(PeExport{name, base + func_rva});
    }
    return true;
}

void ExportCounters::configure(std::string module, std::vector<PeExport> exports) {
    exports.erase(std::remove_if(exports.begin(), exports.end(),
                                 [](const PeExport& e) { return e.address == 0; }),
                  exports.end());
    std::stable_sort(exports.begin(), exports.end(),
                     [](const PeExport& a, const PeExport& b) { return a.address < b.address; });
    // Two names for one address share the first name's counter.
    exports.erase(std::unique(exports.begin(), exports.end(),
                              [](const PeExport& a, const PeExport& b) { return a.address == b.address; }),
                  exports.end());
    module_ = std::move(module);
    entries_ = std::move(exports);
    counts_.reset(entries_.empty() ? nullptr : new std::atomic<uint64_t>[entries_.size()]());
    min_ = entries_.empty() ? 0 : entries_.front().address;
    max_ = entries_.empty() ? 0 : entries_.back().address;
    count_ = entries_.size();
}

void ExportCounters::clear() {
    count_ = 0;
    min_ = max_ = 0;
    entries_.clear();
    counts_.reset();
    module_.clear();
}

long ExportCounters::hit(uint64_t target, bool* first) noexcept {
    if (count_ == 0 || target < min_ || target > max_) {
        return -1;
    }
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), target,
                                     [](const PeExport& e, uint64_t t) { return e.address < t; });
    if (it == entries_.end() || it->address != target) {
        return -1;
    }
    const auto index = static_cast<size_t>(it - entries_.begin());
    const uint64_t before = counts_[index].fetch_add(1, std::memory_order_relaxed);
    if (first != nullptr) {
        *first = before == 0;
    }
    return static_cast<long>(index);
}

long ExportCounters::find(std::string_view export_name) const {
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].name == export_name) {
            return static_cast<long>(i);
        }
    }
    return -1;
}

}  // namespace playgta6::guest
