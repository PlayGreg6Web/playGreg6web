#include "guest/export_thunks.h"

#include <mutex>
#include <string>
#include <unordered_map>

#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace playgta6::guest {

namespace {

struct ReasonRegistry {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::string> by_proc;
};

ReasonRegistry& reasons() {
    static ReasonRegistry registry;
    return registry;
}

}  // namespace

ExportThunkStats register_export_thunks(std::span<const BootExport> table) {
    namespace win32 = playgta6::os::win32;
    ExportThunkStats stats;
    stats.rows = table.size();
    const uint32_t saved_error = win32::GetLastError();
    for (const BootExport& row : table) {
        if (row.dll == nullptr || row.name == nullptr) {
            ++stats.unregistered;
            continue;
        }
        const uint64_t proc = win32::resolve_import(row.dll, row.name);
        if (proc == 0) {
            ++stats.unregistered;
            continue;
        }
        if (guest_callable_for_proc(proc) != 0) {
            ++stats.shared;
            continue;
        }
        if (row.thunk == nullptr) {
            ++stats.unavailable;
            if (row.reason != nullptr) {
                ReasonRegistry& reg = reasons();
                std::lock_guard<std::mutex> lock(reg.mutex);
                reg.by_proc.emplace(proc, row.reason);
            }
            continue;
        }
        if (register_host_callable(row.thunk, row.name, proc) == 0) {
            ++stats.failed;
            break;  // table full: nothing more fits
        }
        ++stats.registered;
    }
    win32::SetLastError(saved_error);
    return stats;
}

const char* unavailable_reason_for_proc(uint64_t proc_address) {
    ReasonRegistry& reg = reasons();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const auto it = reg.by_proc.find(proc_address);
    return it != reg.by_proc.end() ? it->second.c_str() : nullptr;
}

void reset_export_thunks() {
    ReasonRegistry& reg = reasons();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.by_proc.clear();
}

}  // namespace playgta6::guest

extern "C" size_t pg6_guest_register_export_thunks(const Pg6BootExport* table, size_t count) {
    if (table == nullptr) {
        return 0;
    }
    return playgta6::guest::register_export_thunks(std::span<const Pg6BootExport>(table, count))
        .registered;
}
