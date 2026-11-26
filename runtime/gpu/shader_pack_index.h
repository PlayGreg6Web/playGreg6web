#pragma once

// Shader pack index (G47): which shader identities a WGSL pack (.wgsl.pack, "WGPK", read on the
// WebGPU side by web/workers/wgsl_pack_loader.js) can resolve. The native layer never compiles
// WGSL: it only needs the keys, to decide which pipeline stages are resolved and to emit
// CreateShaderModule packets whose code is the hash-only string "0x%016x", which the WebGPU worker
// resolves from its loaded pack.
//
// WGPK layout (little endian): header {char magic[4] "WGPK", u32 version (1), u32 entry_count,
// u32 flags}, then entry_count entries {u64 hash, u8 stage (1 vertex, 2 fragment, 3 compute),
// u8 reserved[3], u32 offset, u32 length} (20 bytes each); payloads are UTF-8 WGSL.
//
// WGPK v2 (packs larger than 4 GiB): identical 16-byte header with version == 2; entries are 24
// bytes {u64 hash, u8 stage, u8 reserved[3], u32 length, u64 offset}. Per-shader payloads stay
// below 4 GiB (u32 length) while the file offset is 64-bit. v1 packs stay valid unchanged.

#include <cstddef>
#include <cstdint>
#include <string>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace playgta6::gpu {

inline constexpr size_t kShaderPackHeaderBytes = 16;
inline constexpr size_t kShaderPackEntryBytes = 20;     // v1 entry size
inline constexpr size_t kShaderPackEntryBytesV2 = 24;    // v2 entry size (64-bit offset)

// Entry record size for a WGPK version (20 for v1, 24 for v2); 0 for unsupported versions.
size_t shader_pack_entry_bytes(uint32_t version);

enum class PackStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 3 };

// Pack key -> stage. A key present for several stages keeps the first.
using ShaderPackIndex = std::unordered_map<uint64_t, PackStage>;

// Parses and validates the 16-byte WGPK header ("WGPK" magic, version 1 or 2) and writes the entry
// count to `out_count` and, when non-null, the version to `out_version`. False + message when
// malformed.
bool parse_shader_pack_header(const uint8_t* data,
                              size_t size,
                              uint32_t* out_count,
                              std::string* error,
                              uint32_t* out_version = nullptr);

// Parses the header and entry table from a TOC-only (or full-file) buffer without reading WGSL
// payloads. Validates magic, version 1/2, `toc_size >= 16 + count * entry_bytes(version)`, entry stages in 1..3,
// non-overflowing `offset + length` (32-bit for v1, 64-bit for v2), and `offset >= toc bytes` (unless `length == 0`).
// When `total_file_size > 0`, also validates `uint64_t{offset} + length <= total_file_size`.
bool parse_shader_pack_index_toc(const uint8_t* toc_data,
                                 size_t toc_size,
                                 uint64_t total_file_size,
                                 ShaderPackIndex* out,
                                 std::string* error);

// Parses the header and entry table of a complete in-memory pack (payloads are not read).
// Equivalent to `parse_shader_pack_index_toc(data, size, size, out, error)`.
bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error);

// The hash-only CreateShaderModule code for a pack key ("0x" + 16 lowercase hex digits).
std::string pack_key_code(uint64_t key);

// Deterministic CreateShaderModule ids for pack modules. A pack key's id is kPackModuleIdBase + its rank
// among the pack's keys in ascending order, so it does not depend on which thread (the pipeline prewarm
// thread or the title) reaches the module first: captures of two runs carry the same module ids.
// Ids below kPackModuleIdBase belong to the Device's sequential allocator.
inline constexpr uint32_t kPackModuleIdBase = 0x40000000u;

class PackModuleTable {
public:
    explicit PackModuleTable(const ShaderPackIndex& pack);
    // Id of `key`'s module, building it with `create(id)` exactly once (the first caller runs it, a
    // concurrent caller waits and gets the same id). 0 when `key` is not a pack key.
    uint32_t get_or_create(uint64_t key, const std::function<void(uint32_t id)>& create);
    // Deterministic id of `key` without creating it; 0 when not a pack key.
    uint32_t id_of(uint64_t key) const;
    size_t created() const;

private:
    std::unordered_map<uint64_t, uint32_t> ids_;
    mutable std::mutex mutex_;
    std::unordered_set<uint64_t> built_;
};

}  // namespace playgta6::gpu
