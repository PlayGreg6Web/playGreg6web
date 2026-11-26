#pragma once
// Per-export call counters for a lifted DLL (B2: Bink2Scarlett.dll).
//
// A guest-to-guest call into a lifted DLL never crosses an import, so import-name milestones
// cannot see it. The dispatch observer sees every indirect call target instead; this table maps
// a target address to one of the DLL's real exported entry points and counts it. The hit path is
// a range check plus a binary search over a handful of addresses, no locks.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace playgta6::guest {

struct PeExport {
    std::string name;
    uint64_t address = 0;  // guest VA (image base + RVA)
};

// Named, non-forwarded exports of a mapped PE64 image, read through `read(va, dst, len)`.
// Returns false when the headers or the export directory are unreadable or implausible.
using ExportReadFn = std::function<bool(uint64_t va, void* dst, size_t len)>;
bool enumerate_pe_exports(const ExportReadFn& read, uint64_t image_base, std::vector<PeExport>& out);

class ExportCounters {
public:
    // Replaces the table (call before guest code runs). Entries with address 0 are dropped.
    void configure(std::string module, std::vector<PeExport> exports);
    void clear();
    bool active() const { return count_ != 0; }

    // Counts a call to `target`. Returns the export's index on a hit, -1 otherwise.
    // `first` is set when this was the export's first call.
    long hit(uint64_t target, bool* first = nullptr) noexcept;

    size_t size() const { return count_; }
    const std::string& module() const { return module_; }
    const std::string& name(size_t i) const { return entries_[i].name; }
    uint64_t address(size_t i) const { return entries_[i].address; }
    uint64_t calls(size_t i) const { return counts_[i].load(std::memory_order_relaxed); }
    long find(std::string_view export_name) const;

private:
    std::string module_;
    std::vector<PeExport> entries_;  // sorted by address
    std::unique_ptr<std::atomic<uint64_t>[]> counts_;
    size_t count_ = 0;
    uint64_t min_ = 0;
    uint64_t max_ = 0;
};

}  // namespace playgta6::guest
