#pragma once

// BCPack: the lossless packing of BC block streams that the console's DirectStorage hardware
// unpacks (DSTORAGE_REQUEST_OPTIONS bits 1-4 name the BC format: 1 BC1, 3 BC3, 4 BC4, 5 BC5,
// 6 BC6H, 7 BC7). Decode only; BC2 is not supported. See docs/BCPACK.md.
//
// The algorithm lives here; the format's constant tables (rANS models, lane layouts, index rank
// tables, BC7 control words, per-format dictionaries) do not. They are loaded at run time from a
// file the user supplies (`--bcpack-tables <file>` / PLAYGTA6_BCPACK_TABLES) in the documented
// container below. The repository never ships that file, and no tool here produces it. Without it
// BCPack requests keep failing with E_NOTIMPL.
//
// Stream outline (no header, no size): 12 or 16 interleaved rANS lanes. Each lane is a run of
// fragments (a header byte naming one of 128 static models, a length, the payload); a fixed
// round-robin schedule hands the lanes at most 8 input bytes per round. The lanes' symbols form
// one record per block: two back references (one of the last 256 blocks, or a per-format
// dictionary entry) and the residuals of the endpoints and indices against them.
//
// Table file (all integers little-endian):
//   header   char magic[8] = "PG6BCPAK", u32 version = 1, u32 section_count
//   sections section_count x { char tag[4], u32 version = 1, u32 offset, u32 size }
//   tags     MODL (required), LANE (required), RANK, BC7C, DIC1, DIC3, DIC4, DIC5, DIC6, DIC7.
//            Unknown tags are ignored. Payload layouts: docs/BCPACK.md.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace playgta6::os::bcpack {

struct Tables;  // parsed table file (opaque)

// Parses a table file. On failure returns nullptr and says why in *error.
std::shared_ptr<const Tables> parse_tables(const uint8_t* data, size_t size, std::string* error);

// True if the tables carry what BC format `bc` needs (its lane layout and the sections it uses).
bool supports(const Tables& tables, uint32_t bc);

// Comma-separated BC formats the tables support, e.g. "BC1, BC4, BC7" (empty if none).
std::string supported_formats(const Tables& tables);

enum class DecodeStatus {
    Ok,
    Unsupported,  // BC format not decodable with these tables
    BadSize,      // destination not a whole, non-zero number of blocks
    Truncated,    // the stream ran out before every block was decoded (or is corrupt)
    Corrupt,      // a record names something that does not exist (e.g. an unused BC7 control code)
};
const char* decode_status_name(DecodeStatus status);

// Unpacks `src` (BC format `bc`) into `dst`; dst_size gives the block count. Bytes after the last
// one the lanes need are ignored.
DecodeStatus decode(const Tables& tables, uint32_t bc, const uint8_t* src, size_t src_size, uint8_t* dst,
                    size_t dst_size);

// Bytes per block of BC format `bc` (8 or 16), 0 for an unknown format.
size_t block_bytes(uint32_t bc);

// Process-wide tables used by DirectStorage. load_tables_file() reads and parses `path`;
// set_active_tables(nullptr) unloads them.
bool load_tables_file(const std::string& path, std::string* error);
void set_active_tables(std::shared_ptr<const Tables> tables);
std::shared_ptr<const Tables> active_tables();

// Test hook: runs only the lane stage for `blocks` blocks of format `bc`, taking each input byte
// from fetch(context, lane, &byte) (false: no more input), and stores the per-block records
// (lane_count * symbols_per_block symbols each). Returns false if the input ends early.
using FetchByte = bool (*)(void* context, size_t lane, uint8_t* byte);
bool decode_lanes(const Tables& tables, uint32_t bc, size_t blocks, FetchByte fetch, void* context,
                  uint8_t* records);

}  // namespace playgta6::os::bcpack
