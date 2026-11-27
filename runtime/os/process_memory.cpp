#include "os/process_memory.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#include <psapi.h>
#endif

namespace playgta6::os {

bool process_memory_bytes(uint64_t* private_bytes, uint64_t* working_set_bytes, uint64_t* peak_private_bytes) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        return false;
    }
    *private_bytes = pmc.PrivateUsage;
    *working_set_bytes = pmc.WorkingSetSize;
    if (peak_private_bytes != nullptr) {
        *peak_private_bytes = pmc.PeakPagefileUsage;
    }
    return true;
#else
    (void)private_bytes;
    (void)working_set_bytes;
    (void)peak_private_bytes;
    return false;
#endif
}

uint64_t available_physical_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullAvailPhys : 0;
#else
    return 0;
#endif
}

uint64_t dedicated_video_memory_bytes() {
#ifdef _WIN32
    // Loaded at run time: no link dependency on dxgi.lib.
    HMODULE dxgi = LoadLibraryW(L"dxgi");  // the loader appends .dll
    if (dxgi == nullptr) return 0;
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    auto create = reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    uint64_t best = 0;
    IDXGIFactory1* factory = nullptr;
    if (create != nullptr && SUCCEEDED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        IDXGIAdapter1* adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            if (SUCCEEDED(adapter->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                best = std::max<uint64_t>(best, d.DedicatedVideoMemory);
            }
            adapter->Release();
        }
        factory->Release();
    }
    FreeLibrary(dxgi);
    return best;
#else
    return 0;
#endif
}

#ifdef _WIN32
namespace {
// Pages of [base, base + size) in the working set.
uint64_t resident_bytes(uint64_t base, uint64_t size) {
    constexpr size_t kBatch = 1u << 16;
    static thread_local std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(kBatch);
    uint64_t pages = 0;
    for (uint64_t off = 0; off < size; off += uint64_t{kBatch} * 4096) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(kBatch, (size - off + 4095) / 4096));
        for (size_t i = 0; i < n; ++i) {
            info[i].VirtualAddress = reinterpret_cast<void*>(static_cast<uintptr_t>(base + off + i * 4096));
        }
        if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(), static_cast<DWORD>(n * sizeof(info[0])))) {
            break;
        }
        for (size_t i = 0; i < n; ++i) {
            pages += info[i].VirtualAttributes.Valid ? 1 : 0;
        }
    }
    return pages * 4096;
}
}  // namespace
#endif

bool process_commit_map(uint64_t arena_lo, uint64_t arena_hi, ProcessCommitMap* out) {
#ifdef _WIN32
    std::unordered_map<uint64_t, uint64_t> by_base;  // AllocationBase -> private commit outside the arena
    uint8_t* p = nullptr;
    MEMORY_BASIC_INFORMATION mbi;
    while (VirtualQuery(p, &mbi, sizeof(mbi)) == sizeof(mbi)) {
        const uint64_t b = reinterpret_cast<uintptr_t>(mbi.BaseAddress);
        const uint64_t n = mbi.RegionSize;
        if (mbi.State == MEM_COMMIT) {
            const bool in_arena = b >= arena_lo && b < arena_hi;
            if (mbi.Type == MEM_IMAGE) {
                out->image += n;
            } else if (mbi.Type == MEM_MAPPED) {
                (in_arena ? out->arena_mapped : out->other_mapped) += n;
            } else if (in_arena) {
                out->arena_private += n;
                out->arena_private_resident += resident_bytes(b, n);
            } else {
                out->other_private += n;
                out->other_private_resident += resident_bytes(b, n);
                by_base[reinterpret_cast<uintptr_t>(mbi.AllocationBase)] += n;
            }
        }
        const uint64_t next = b + n;
        if (next <= b) {
            break;
        }
        p = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(next));
    }
    {
        std::unordered_map<uint64_t, uint64_t> by_size;
        for (const auto& kv : by_base) {
            if (kv.second >= (uint64_t{4} << 20)) {
                ++by_size[kv.second];
            }
        }
        std::vector<std::pair<uint64_t, uint64_t>> sv(by_size.begin(), by_size.end());
        std::sort(sv.begin(), sv.end(), [](const auto& a, const auto& c) {
            return a.first * a.second > c.first * c.second;
        });
        for (int i = 0; i < ProcessCommitMap::kSizes && i < static_cast<int>(sv.size()); ++i) {
            out->size_bytes[i] = sv[static_cast<size_t>(i)].first;
            out->size_count[i] = sv[static_cast<size_t>(i)].second;
        }
    }
    std::vector<std::pair<uint64_t, uint64_t>> v(by_base.begin(), by_base.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return a.second > c.second; });
    for (int i = 0; i < ProcessCommitMap::kTop && i < static_cast<int>(v.size()); ++i) {
        out->top_base[i] = v[static_cast<size_t>(i)].first;
        out->top_bytes[i] = v[static_cast<size_t>(i)].second;
    }
    return true;
#else
    (void)arena_lo;
    (void)arena_hi;
    (void)out;
    return false;
#endif
}

}  // namespace playgta6::os
