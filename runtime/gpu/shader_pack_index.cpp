#include "gpu/shader_pack_index.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

namespace playgta6::gpu {

namespace {

uint32_t u32_at(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

uint64_t u64_at(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

void set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

}  // namespace

size_t shader_pack_entry_bytes(uint32_t version) {
    return version == 1 ? kShaderPackEntryBytes : version == 2 ? kShaderPackEntryBytesV2 : 0;
}

bool parse_shader_pack_header(const uint8_t* data,
                              size_t size,
                              uint32_t* out_count,
                              std::string* error,
                              uint32_t* out_version) {
    if (data == nullptr || size < kShaderPackHeaderBytes || std::memcmp(data, "WGPK", 4) != 0) {
        set_error(error, "not a WGPK shader pack");
        return false;
    }
    const uint32_t version = u32_at(data + 4);
    const uint32_t count = u32_at(data + 8);
    if (shader_pack_entry_bytes(version) == 0) {
        set_error(error, "unsupported WGPK version " + std::to_string(version));
        return false;
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    if (out_version != nullptr) {
        *out_version = version;
    }
    return true;
}

bool parse_shader_pack_index_toc(const uint8_t* toc_data,
                                 size_t toc_size,
                                 uint64_t total_file_size,
                                 ShaderPackIndex* out,
                                 std::string* error) {
    if (out == nullptr) {
        set_error(error, "null shader pack index output");
        return false;
    }
    out->clear();

    uint32_t count = 0;
    uint32_t version = 0;
    if (!parse_shader_pack_header(toc_data, toc_size, &count, error, &version)) {
        return false;
    }
    const size_t entry_bytes = shader_pack_entry_bytes(version);
    if (count > (toc_size - kShaderPackHeaderBytes) / entry_bytes) {
        set_error(error, "WGPK entry table exceeds the file");
        return false;
    }
    const uint64_t toc_bytes =
        uint64_t{kShaderPackHeaderBytes} + uint64_t{count} * uint64_t{entry_bytes};
    if (total_file_size > 0 && toc_bytes > total_file_size) {
        set_error(error, "WGPK entry table exceeds the file");
        return false;
    }

    out->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = toc_data + kShaderPackHeaderBytes + size_t{i} * entry_bytes;
        uint64_t hash;
        std::memcpy(&hash, e, 8);
        const uint8_t stage = e[8];
        const uint64_t offset = version == 2 ? u64_at(e + 16) : uint64_t{u32_at(e + 12)};
        const uint32_t length = u32_at(e + (version == 2 ? 12 : 16));
        if (stage < 1 || stage > 3) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " has stage " + std::to_string(stage));
            return false;
        }
        if (offset > std::numeric_limits<uint64_t>::max() - length) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload range overflows");
            return false;
        }
        const uint64_t payload_end = offset + uint64_t{length};
        if (version == 1 && payload_end > std::numeric_limits<uint32_t>::max()) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload range overflows");
            return false;
        }
        if (length != 0 && offset < toc_bytes) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload overlaps header/TOC");
            return false;
        }
        if (total_file_size > 0 && payload_end > total_file_size) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload exceeds the file");
            return false;
        }
        out->emplace(hash, static_cast<PackStage>(stage));
    }
    return true;
}

bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error) {
    return parse_shader_pack_index_toc(data, size, static_cast<uint64_t>(size), out, error);
}

std::string pack_key_code(uint64_t key) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016" PRIx64, key);
    return b;
}

PackModuleTable::PackModuleTable(const ShaderPackIndex& pack) {
    std::vector<uint64_t> keys;
    keys.reserve(pack.size());
    for (const auto& kv : pack) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    ids_.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        ids_.emplace(keys[i], kPackModuleIdBase + static_cast<uint32_t>(i));
    }
}

uint32_t PackModuleTable::id_of(uint64_t key) const {
    auto it = ids_.find(key);
    return it == ids_.end() ? 0u : it->second;
}

uint32_t PackModuleTable::get_or_create(uint64_t key, const std::function<void(uint32_t id)>& create) {
    const uint32_t id = id_of(key);
    if (id == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);  // held across create: a concurrent caller waits for the one build
    if (built_.insert(key).second) {
        create(id);
    }
    return id;
}

size_t PackModuleTable::created() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return built_.size();
}

}  // namespace playgta6::gpu
