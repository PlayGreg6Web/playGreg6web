#include "gpu/shader_pack_index.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace playgta6::gpu {

namespace {

constexpr size_t kHeaderBytes = 16;
constexpr size_t kEntryBytes = 20;

uint32_t u32_at(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

}  // namespace

bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error) {
    out->clear();
    if (data == nullptr || size < kHeaderBytes || std::memcmp(data, "WGPK", 4) != 0) {
        *error = "not a WGPK shader pack";
        return false;
    }
    const uint32_t version = u32_at(data + 4);
    const uint32_t count = u32_at(data + 8);
    if (version != 1) {
        *error = "unsupported WGPK version " + std::to_string(version);
        return false;
    }
    if (count > (size - kHeaderBytes) / kEntryBytes) {
        *error = "WGPK entry table exceeds the file";
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = data + kHeaderBytes + size_t{i} * kEntryBytes;
        uint64_t hash;
        std::memcpy(&hash, e, 8);
        const uint8_t stage = e[8];
        const uint32_t offset = u32_at(e + 12);
        const uint32_t length = u32_at(e + 16);
        if (stage < 1 || stage > 3) {
            *error = "WGPK entry " + std::to_string(i) + " has stage " + std::to_string(stage);
            return false;
        }
        if (uint64_t{offset} + length > size) {
            *error = "WGPK entry " + std::to_string(i) + " payload exceeds the file";
            return false;
        }
        out->emplace(hash, static_cast<PackStage>(stage));
    }
    return true;
}

std::string pack_key_code(uint64_t key) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016" PRIx64, key);
    return b;
}

}  // namespace playgta6::gpu
