// Native executor: which WGSL texts may be released after their module and scans exist (#2074).
// A pack-backed shader's text is ~3 MB (~2 GB over a std-pack boot); the pack can deliver it again, so
// it is not kept. Inline WGSL (hash 0) has no other source, and a pack miss compiled the tiny hash text.
#pragma once

#include <cstdint>

namespace playgta6::gpu::native_exec {

inline bool shader_text_droppable(uint64_t pack_hash, bool pack_miss) { return pack_hash != 0 && !pack_miss; }

}  // namespace playgta6::gpu::native_exec
