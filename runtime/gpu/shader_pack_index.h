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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace playgta6::gpu {

enum class PackStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 3 };

// Pack key -> stage. A key present for several stages keeps the first.
using ShaderPackIndex = std::unordered_map<uint64_t, PackStage>;

// Parses the header and entry table (payloads are not read). False + message when malformed.
bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error);

// The hash-only CreateShaderModule code for a pack key ("0x" + 16 lowercase hex digits).
std::string pack_key_code(uint64_t key);

}  // namespace playgta6::gpu
