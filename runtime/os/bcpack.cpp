#include "os/bcpack.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace playgta6::os::bcpack {

namespace {

constexpr uint32_t kModels = 128;
constexpr uint32_t kMaxBuckets = 9;      // a model of group g uses g + 2 buckets
constexpr uint32_t kHistoryBlocks = 256;  // back references reach this many blocks
constexpr uint32_t kDictionaryEntries = 255;
constexpr uint32_t kRansFloor = 1u << 24;  // a lane decodes above this state, reads below it
constexpr uint32_t kRoundBytes = 8;        // input bytes per schedule round, all lanes together
constexpr uint32_t kMaxLanes = 16;
constexpr uint32_t kBc7ControlWords = 512;
constexpr uint32_t kFormats = 8;  // indexed by BC number

// Symbols per block record, by BC number (the decoder below reads exactly these positions).
constexpr size_t record_symbols(uint32_t bc) {
    switch (bc) {
        case 1:
        case 4:
            return 12;
        case 3:
        case 5:
            return 24;
        case 7:
            return 32;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

struct Model {
    uint32_t used = 0;                   // buckets in use
    uint32_t freq[kMaxBuckets] = {};     // per-symbol frequency of each bucket
    uint32_t first[kMaxBuckets + 1] = {};  // first slot of each bucket; first[used] = 65536
};

struct LaneLayout {
    bool present = false;
    uint8_t lanes = 0;
    uint8_t per_block = 0;  // symbols each lane contributes per block (1 or 2)
    uint8_t width[kMaxLanes] = {};
};

// Endpoints and indices of BC1, BC3, BC4 and BC5 as small unsigned values.
struct PlainBlock {
    uint8_t ep[8] = {};
    uint8_t ix[32] = {};
};

struct Bc7Block {
    uint16_t control = 0;
    uint16_t rows[3][8] = {};  // per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned to 16 bits
    uint8_t index_bits[2] = {};
    int16_t index[2][16] = {};
};

}  // namespace

struct Tables {
    Model models[kModels];
    LaneLayout layouts[kFormats];
    bool has_ranks = false;
    uint8_t rank_bc1[256] = {};
    uint8_t rank_bc3_alpha[64] = {};
    uint8_t rank_bc3_color[256] = {};
    uint8_t rank_bc4[64] = {};
    uint8_t rank_bc5[64] = {};
    bool has_bc7_control = false;
    uint16_t bc7_control[kBc7ControlWords] = {};
    std::vector<PlainBlock> plain_dictionary[kFormats];
    std::vector<Bc7Block> bc7_dictionary;
};

namespace {

class ByteCursor {
public:
    ByteCursor(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
    uint8_t u8() { return p_ < end_ ? *p_++ : 0; }
    uint16_t u16() {
        const uint16_t lo = u8();
        return static_cast<uint16_t>(lo | (u8() << 8));
    }
    uint32_t u32() {
        const uint32_t lo = u16();
        return lo | (static_cast<uint32_t>(u16()) << 16);
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

constexpr char kMagic[8] = {'P', 'G', '6', 'B', 'C', 'P', 'A', 'K'};
constexpr uint32_t kFileVersion = 1;
constexpr uint32_t kSectionVersion = 1;

struct Section {
    const uint8_t* data = nullptr;
    uint32_t size = 0;
    bool found = false;
};

// Endpoint and index element counts of a plain format's dictionary entry.
struct PlainShape {
    uint32_t endpoints;
    uint32_t indices;
};
constexpr PlainShape plain_shape(uint32_t bc) {
    switch (bc) {
        case 1:
            return {6, 16};
        case 3:
            return {8, 32};
        case 4:
            return {2, 16};
        case 5:
            return {4, 32};
        default:
            return {0, 0};
    }
}
constexpr size_t kBc7EntryBytes = 2 + 3 * 8 * 2 + 2 + 2 * 16 * 2;

bool parse_models(const Section& s, Tables* t, std::string* error) {
    if (s.size != kModels * kMaxBuckets * 2) {
        *error = "MODL section must hold 128 x 9 u16 frequencies (" + std::to_string(kModels * kMaxBuckets * 2) +
                 " bytes), not " + std::to_string(s.size);
        return false;
    }
    ByteCursor in(s.data, s.size);
    for (uint32_t m = 0; m < kModels; ++m) {
        Model& model = t->models[m];
        model.used = m / 16 + 2;
        uint32_t slot = 0;
        for (uint32_t b = 0; b < kMaxBuckets; ++b) {
            model.freq[b] = in.u16();
        }
        for (uint32_t b = 0; b < model.used; ++b) {
            if (model.freq[b] == 0) {
                *error = "MODL: model " + std::to_string(m) + " has an empty bucket " + std::to_string(b);
                return false;
            }
            model.first[b] = slot;
            // Bucket 0 holds symbol 0; bucket b > 0 holds the 2^(b-1) symbols [2^(b-1), 2^b).
            slot += model.freq[b] << (b == 0 ? 0 : b - 1);
        }
        if (slot != 65536) {
            *error = "MODL: model " + std::to_string(m) + " covers " + std::to_string(slot) +
                     " slots instead of 65536";
            return false;
        }
        for (uint32_t b = model.used; b <= kMaxBuckets; ++b) {
            model.first[b] = 65536;
        }
    }
    return true;
}

bool parse_layouts(const Section& s, Tables* t, std::string* error) {
    ByteCursor in(s.data, s.size);
    const uint32_t count = s.size >= 4 ? in.u32() : 0;
    if (s.size < 4 || s.size != 4 + 20ull * count) {
        *error = "LANE section must be a u32 count and count x 20-byte entries";
        return false;
    }
    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t bc = in.u8();
        LaneLayout layout;
        layout.lanes = in.u8();
        layout.per_block = in.u8();
        in.u8();  // reserved
        for (uint8_t& w : layout.width) {
            w = in.u8();
        }
        const std::string which = "LANE entry " + std::to_string(k) + " (BC" + std::to_string(bc) + ")";
        if (record_symbols(bc) == 0) {
            continue;  // a format this decoder does not handle (e.g. BC6H): ignored
        }
        if (layout.lanes == 0 || layout.lanes > kMaxLanes || (layout.per_block != 1 && layout.per_block != 2) ||
            static_cast<size_t>(layout.lanes) * layout.per_block != record_symbols(bc)) {
            *error = which + ": " + std::to_string(layout.lanes) + " lanes x " + std::to_string(layout.per_block) +
                     " symbols do not make the " + std::to_string(record_symbols(bc)) + "-symbol record";
            return false;
        }
        for (uint32_t j = 0; j < layout.lanes; ++j) {
            // The header's top two bits lower the width by up to 3; the group must stay within 0..7.
            if (layout.width[j] < 3 || layout.width[j] > 7) {
                *error = which + ": lane " + std::to_string(j) + " width " + std::to_string(layout.width[j]) +
                         " is outside 3..7";
                return false;
            }
        }
        layout.present = true;
        t->layouts[bc] = layout;
    }
    return true;
}

bool parse_ranks(const Section& s, Tables* t, std::string* error) {
    constexpr size_t kSize = 256 + 64 + 256 + 64 + 64;
    if (s.size != kSize) {
        *error = "RANK section must be " + std::to_string(kSize) + " bytes";
        return false;
    }
    const uint8_t* p = s.data;
    for (auto [dst, n] : {std::pair<uint8_t*, size_t>{t->rank_bc1, 256}, {t->rank_bc3_alpha, 64},
                          {t->rank_bc3_color, 256}, {t->rank_bc4, 64}, {t->rank_bc5, 64}}) {
        std::memcpy(dst, p, n);
        p += n;
    }
    t->has_ranks = true;
    return true;
}

bool parse_bc7_control(const Section& s, Tables* t, std::string* error) {
    if (s.size != kBc7ControlWords * 2) {
        *error = "BC7C section must be 512 u16 control words";
        return false;
    }
    ByteCursor in(s.data, s.size);
    for (uint16_t& c : t->bc7_control) {
        c = in.u16();
    }
    t->has_bc7_control = true;
    return true;
}

bool parse_plain_dictionary(uint32_t bc, const Section& s, Tables* t, std::string* error) {
    const PlainShape shape = plain_shape(bc);
    const size_t entry = shape.endpoints + shape.indices;
    if (s.size != kDictionaryEntries * entry) {
        *error = "DIC" + std::to_string(bc) + " section must be 255 x " + std::to_string(entry) + " bytes";
        return false;
    }
    std::vector<PlainBlock>& dict = t->plain_dictionary[bc];
    dict.resize(kDictionaryEntries);
    const uint8_t* p = s.data;
    for (PlainBlock& b : dict) {
        std::memcpy(b.ep, p, shape.endpoints);
        std::memcpy(b.ix, p + shape.endpoints, shape.indices);
        p += entry;
    }
    return true;
}

bool parse_bc7_dictionary(const Section& s, Tables* t, std::string* error) {
    if (s.size != kDictionaryEntries * kBc7EntryBytes) {
        *error = "DIC7 section must be 255 x " + std::to_string(kBc7EntryBytes) + " bytes";
        return false;
    }
    ByteCursor in(s.data, s.size);
    t->bc7_dictionary.resize(kDictionaryEntries);
    for (Bc7Block& b : t->bc7_dictionary) {
        b.control = in.u16();
        for (auto& row : b.rows) {
            for (uint16_t& v : row) {
                v = in.u16();
            }
        }
        b.index_bits[0] = in.u8();
        b.index_bits[1] = in.u8();
        for (auto& set : b.index) {
            for (int16_t& v : set) {
                v = static_cast<int16_t>(in.u16());
            }
        }
    }
    return true;
}

}  // namespace

std::shared_ptr<const Tables> parse_tables(const uint8_t* data, size_t size, std::string* error) {
    std::string scratch;
    std::string& why = error != nullptr ? *error : scratch;
    if (data == nullptr || size < 16 || std::memcmp(data, kMagic, sizeof(kMagic)) != 0) {
        why = "not a BCPack table file (magic \"PG6BCPAK\" missing)";
        return nullptr;
    }
    ByteCursor head(data + 8, 8);
    const uint32_t version = head.u32();
    const uint32_t count = head.u32();
    if (version != kFileVersion) {
        why = "table file version " + std::to_string(version) + " is not supported (expected 1)";
        return nullptr;
    }
    if (count > 4096 || 16 + 16ull * count > size) {
        why = "section directory runs past the end of the file";
        return nullptr;
    }
    struct Known {
        const char* tag;
        Section section;
    };
    std::array<Known, 9> known = {{{"MODL", {}},
                                   {"LANE", {}},
                                   {"RANK", {}},
                                   {"BC7C", {}},
                                   {"DIC1", {}},
                                   {"DIC3", {}},
                                   {"DIC4", {}},
                                   {"DIC5", {}},
                                   {"DIC7", {}}}};
    ByteCursor dir(data + 16, 16 * static_cast<size_t>(count));
    for (uint32_t k = 0; k < count; ++k) {
        char tag[4];
        for (char& c : tag) {
            c = static_cast<char>(dir.u8());
        }
        const uint32_t section_version = dir.u32();
        const uint32_t offset = dir.u32();
        const uint32_t length = dir.u32();
        for (Known& entry : known) {
            if (std::memcmp(entry.tag, tag, 4) != 0) {
                continue;
            }
            const std::string name(tag, 4);
            if (entry.section.found) {
                why = "section " + name + " appears twice";
                return nullptr;
            }
            if (section_version != kSectionVersion) {
                why = "section " + name + " version " + std::to_string(section_version) +
                      " is not supported (expected 1)";
                return nullptr;
            }
            if (static_cast<uint64_t>(offset) + length > size) {
                why = "section " + name + " runs past the end of the file";
                return nullptr;
            }
            entry.section = {data + offset, length, true};
        }
    }
    auto t = std::make_shared<Tables>();
    auto section = [&](const char* tag) -> const Section& {
        for (const Known& entry : known) {
            if (std::memcmp(entry.tag, tag, 4) == 0) {
                return entry.section;
            }
        }
        return known[0].section;  // unreachable: every caller names a known tag
    };
    if (!section("MODL").found || !section("LANE").found) {
        why = "the MODL and LANE sections are required";
        return nullptr;
    }
    if (!parse_models(section("MODL"), t.get(), &why) || !parse_layouts(section("LANE"), t.get(), &why)) {
        return nullptr;
    }
    if (section("RANK").found && !parse_ranks(section("RANK"), t.get(), &why)) {
        return nullptr;
    }
    if (section("BC7C").found && !parse_bc7_control(section("BC7C"), t.get(), &why)) {
        return nullptr;
    }
    for (uint32_t bc : {1u, 3u, 4u, 5u}) {
        const char tag[5] = {'D', 'I', 'C', static_cast<char>('0' + bc), 0};
        if (section(tag).found && !parse_plain_dictionary(bc, section(tag), t.get(), &why)) {
            return nullptr;
        }
    }
    if (section("DIC7").found && !parse_bc7_dictionary(section("DIC7"), t.get(), &why)) {
        return nullptr;
    }
    return t;
}

bool supports(const Tables& t, uint32_t bc) {
    if (bc >= kFormats || record_symbols(bc) == 0 || !t.layouts[bc].present) {
        return false;
    }
    if (bc == 7) {
        return t.has_bc7_control && !t.bc7_dictionary.empty();
    }
    return t.has_ranks && !t.plain_dictionary[bc].empty();
}

std::string supported_formats(const Tables& t) {
    std::string out;
    for (uint32_t bc = 1; bc < kFormats; ++bc) {
        if (supports(t, bc)) {
            out += (out.empty() ? "BC" : ", BC") + std::to_string(bc);
        }
    }
    return out;
}

size_t block_bytes(uint32_t bc) {
    switch (bc) {
        case 1:
        case 4:
            return 8;
        case 3:
        case 5:
        case 7:
            return 16;
        default:
            return 0;
    }
}

const char* decode_status_name(DecodeStatus status) {
    switch (status) {
        case DecodeStatus::Ok:
            return "ok";
        case DecodeStatus::Unsupported:
            return "unsupported format";
        case DecodeStatus::BadSize:
            return "destination is not a whole number of blocks";
        case DecodeStatus::Truncated:
            return "stream ends early or is corrupt";
        case DecodeStatus::Corrupt:
            return "corrupt block record";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------

namespace {

struct Lane {
    enum Phase : uint8_t { kHeader, kLengthLow, kLengthHigh, kPayload };
    Phase phase = kHeader;
    uint8_t width = 0;
    uint8_t model = 0;
    bool long_length = false;
    bool clear_state_at_end = false;
    bool wants_byte = true;
    bool has_input = false;
    uint8_t input = 0;
    bool has_symbol = false;
    uint8_t symbol = 0;
    bool blocked = false;
    uint32_t state = 0;
    uint32_t bytes_left = 0;
};

// One rANS decode step on `x` (> 2^24) with `model`: the symbol and the next state.
inline uint8_t rans_decode(const Model& model, uint32_t x, uint32_t* next) {
    const uint32_t slot = x & 0xFFFF;
    uint32_t b = model.used - 1;
    while (model.first[b] > slot) {
        --b;
    }
    uint32_t symbol;
    uint32_t freq;
    uint32_t offset;
    if (b == 0) {
        // Symbol 0 owns slots 1 .. freq[0] - 1; slot 0 is never coded.
        symbol = 0;
        freq = model.freq[0] - 1;
        offset = slot - 1;
    } else {
        const uint32_t into = slot - model.first[b];
        const uint32_t q = into / model.freq[b];
        symbol = (1u << (b - 1)) + q;
        freq = model.freq[b];
        offset = into - q * freq;
    }
    *next = (x >> 16) * freq + offset;
    return static_cast<uint8_t>(symbol);
}

// Advances a lane by one step: consumes its input byte as fragment header or length, or, in the
// payload, reads at most one byte into the state and decodes at most one symbol.
inline void step(Lane& lane, const Tables& t) {
    if (lane.phase != Lane::kPayload) {
        if (!lane.has_input) {
            return;
        }
        const uint8_t b = lane.input;
        lane.has_input = false;
        lane.wants_byte = true;
        if (lane.phase == Lane::kHeader) {
            // Bits 0-3: model within the group; 4: clear the state at the end; 5: two length bytes;
            // 6-7: how far the group sits below the lane's width.
            lane.model = static_cast<uint8_t>((lane.width - (b >> 6)) * 16 + (b & 0x0F));
            lane.clear_state_at_end = (b & 0x10) != 0;
            lane.long_length = (b & 0x20) != 0;
            lane.bytes_left = 0;
            lane.phase = Lane::kLengthLow;
        } else if (lane.phase == Lane::kLengthLow && !lane.long_length) {
            lane.bytes_left = b + 1u;
            lane.phase = Lane::kPayload;
        } else if (lane.phase == Lane::kLengthLow) {
            lane.bytes_left = b;
            lane.phase = Lane::kLengthHigh;
        } else {
            lane.bytes_left = ((static_cast<uint32_t>(b) << 8) | lane.bytes_left) + 257u;
            lane.phase = Lane::kPayload;
        }
        return;
    }
    uint32_t x = lane.state;
    if (x < kRansFloor && lane.has_input) {
        lane.has_input = false;
        --lane.bytes_left;
        x = (x << 8) | lane.input;
    }
    lane.wants_byte = lane.bytes_left != 0 && !lane.has_input;
    if (x > kRansFloor) {
        lane.symbol = rans_decode(t.models[lane.model], x, &lane.state);
        lane.has_symbol = true;
        return;
    }
    if (lane.bytes_left != 0) {
        lane.state = x;  // exactly 2^24 stalls the lane; only a lane done with its symbols sits there
        return;
    }
    // End of the fragment. The state carried on is the one from before this step's read.
    lane.phase = Lane::kHeader;
    lane.wants_byte = !lane.has_input;
    if (lane.clear_state_at_end) {
        lane.state = 0;
    }
}

// Runs the schedule until every lane has delivered `blocks * per_block` symbols. Each round the
// lanes step in order; a lane that wants input gets the next byte while the round has budget. A
// lane that already delivered its symbol for the current position holds the next one back.
template <typename Fetch>
bool run_lanes(const Tables& t, const LaneLayout& layout, size_t blocks, Fetch&& fetch, uint8_t* records) {
    std::array<Lane, kMaxLanes> lanes{};
    const uint32_t n = layout.lanes;
    const uint32_t per = layout.per_block;
    for (uint32_t j = 0; j < n; ++j) {
        lanes[j].width = layout.width[j];
    }
    const size_t record_size = static_cast<size_t>(n) * per;
    const uint32_t everyone = (1u << n) - 1u;  // n <= 16
    const size_t positions = blocks * per;
    size_t position = 0;
    uint32_t delivered = 0;
    uint32_t quiet_rounds = 0;
    while (position < positions) {
        uint8_t* out = records + (position / per) * record_size + position % per;
        uint32_t budget_used = 0;
        bool delivered_any = false;
        for (uint32_t j = 0; j < n; ++j) {
            Lane& lane = lanes[j];
            const uint32_t used_before = budget_used;
            if (!lane.blocked) {
                step(lane, t);
            }
            const bool holding = lane.has_symbol && (delivered & (1u << j)) != 0;
            if (lane.wants_byte && !holding && budget_used < kRoundBytes) {
                uint8_t byte = 0;
                if (fetch(j, &byte)) {
                    lane.input = byte;
                    lane.has_input = true;
                    ++budget_used;
                }
            }
            lane.blocked = holding || (lane.wants_byte && used_before >= kRoundBytes);
            if (!lane.blocked && lane.has_symbol) {
                out[j * per] = lane.symbol;
                lane.has_symbol = false;
                delivered |= 1u << j;
                delivered_any = true;
            }
        }
        if (delivered == everyone) {
            ++position;
            delivered = 0;
            delivered_any = true;
        }
        // A valid stream never goes several rounds without reading or delivering anything.
        quiet_rounds = (budget_used != 0 || delivered_any) ? 0 : quiet_rounds + 1;
        if (quiet_rounds > 4) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Block records
// ---------------------------------------------------------------------------

// Zigzag (0, -1, 1, -2, ...) back to a value modulo 2^bits.
inline uint8_t unzigzag_mod(uint8_t v, uint32_t bits) {
    const uint32_t mask = (1u << bits) - 1u;
    const uint32_t magnitude = v >> 1;
    return static_cast<uint8_t>(((v & 1) ? ~magnitude : magnitude) & mask);
}

// Two 4-bit symbols merged bitwise: `even` on bits 0, 2, 4, 6 and `odd` on bits 1, 3, 5, 7.
inline uint8_t merge_nibbles(uint8_t even, uint8_t odd) {
    uint8_t out = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        out |= static_cast<uint8_t>(((even >> k) & 1u) << (2 * k));
        out |= static_cast<uint8_t>(((odd >> k) & 1u) << (2 * k + 1));
    }
    return out;
}

// The record's (endpoint, index) references. The endpoint reference is coded as a zigzag delta
// from the index reference.
struct References {
    uint32_t endpoints;
    uint32_t indices;
};
inline References references(const uint8_t* s, bool four_symbols) {
    const uint8_t delta = four_symbols ? merge_nibbles(s[0], s[1]) : s[0];
    const uint8_t index_ref = four_symbols ? merge_nibbles(s[2], s[3]) : s[1];
    const uint8_t signed_delta = static_cast<uint8_t>((delta >> 1) ^ (0u - (delta & 1u)));
    return {static_cast<uint8_t>(index_ref + signed_delta), index_ref};
}

// Reference `r` of block `i`: 0 is the all-zero block, 1..i earlier blocks, beyond that the
// dictionary (r = i + 1 is entry 0).
template <typename T>
const T& referenced(size_t i, uint32_t r, const std::vector<T>& history, const std::vector<T>& dictionary,
                    const T& zero) {
    if (r == 0) {
        return zero;
    }
    if (r <= i) {
        return history[(i - r) % kHistoryBlocks];
    }
    return dictionary[r - i - 1];
}

// 2-bit index residuals: one symbol per 2x2 quad (quads in raster order), ranked.
void quad_indices(const uint8_t* rank, const uint8_t* symbols, uint8_t* ix) {
    for (uint32_t q = 0; q < 4; ++q) {
        const uint8_t v = rank[symbols[q]];
        const uint32_t corner = (q / 2) * 8 + (q % 2) * 2;
        const uint32_t pixel[4] = {corner, corner + 1, corner + 4, corner + 5};
        for (uint32_t k = 0; k < 4; ++k) {
            ix[pixel[k]] = (v >> (2 * k)) & 3u;
        }
    }
}

// 3-bit index residuals: one symbol per two neighbouring pixels in raster order, ranked.
void pair_indices(const uint8_t* rank, const uint8_t* symbols, size_t count, uint8_t* ix) {
    for (size_t k = 0; k < count; ++k) {
        const uint8_t v = rank[symbols[k] & 63u];
        ix[2 * k] = v & 7u;
        ix[2 * k + 1] = (v >> 4) & 7u;
    }
}

// Position along the endpoint line (0 = first endpoint, top = second) -> BC1/BC4 index code. The
// codes name the endpoints 0 and 1 and the interpolated values 2.. in order from the first.
inline uint64_t line_to_code(uint32_t p, uint32_t top) { return p == 0 ? 0 : p == top ? 1 : p + 1; }

void write_le(uint8_t* out, uint64_t v, size_t bytes) {
    for (size_t k = 0; k < bytes; ++k) {
        out[k] = static_cast<uint8_t>(v >> (8 * k));
    }
}

// BC1 colour half: RGB565 endpoints (5, 6, 5 bit fields) and 2-bit indices.
void pack_color(const uint8_t* ep, const uint8_t* ix, uint8_t* out) {
    const uint64_t c0 = (uint64_t{ep[0]} << 11) | (uint64_t{ep[1]} << 5) | ep[2];
    const uint64_t c1 = (uint64_t{ep[3]} << 11) | (uint64_t{ep[4]} << 5) | ep[5];
    uint64_t bits = 0;
    for (uint32_t p = 0; p < 16; ++p) {
        bits |= line_to_code(ix[p] & 3u, 3) << (2 * p);
    }
    write_le(out, c0 | (c1 << 16) | (bits << 32), 8);
}

// BC4 half: 8-bit endpoints (kept with the top bit flipped) and 3-bit indices.
void pack_alpha(uint8_t e0, uint8_t e1, const uint8_t* ix, uint8_t* out) {
    uint64_t v = static_cast<uint64_t>(e0 ^ 0x80u) | (static_cast<uint64_t>(e1 ^ 0x80u) << 8);
    for (uint32_t p = 0; p < 16; ++p) {
        v |= line_to_code(ix[p] & 7u, 7) << (16 + 3 * p);
    }
    write_le(out, v, 8);
}

struct PlainFormat {
    uint8_t endpoint_bits[8];
    uint8_t index_bits[32];
    bool four_reference_symbols;
};

PlainFormat plain_format(uint32_t bc) {
    PlainFormat f{};
    auto indices = [&f](uint32_t three_bit, uint32_t two_bit) {
        for (uint32_t k = 0; k < three_bit + two_bit; ++k) {
            f.index_bits[k] = k < three_bit ? 3 : 2;
        }
    };
    const uint8_t rgb565x2[6] = {5, 6, 5, 5, 6, 5};
    switch (bc) {
        case 1:
            std::memcpy(f.endpoint_bits, rgb565x2, 6);
            indices(0, 16);
            break;
        case 3:
            std::memcpy(f.endpoint_bits, rgb565x2, 6);
            f.endpoint_bits[6] = f.endpoint_bits[7] = 8;
            indices(16, 16);
            f.four_reference_symbols = true;
            break;
        case 4:
            f.endpoint_bits[0] = f.endpoint_bits[1] = 8;
            indices(16, 0);
            break;
        default:  // 5
            std::fill(f.endpoint_bits, f.endpoint_bits + 4, uint8_t{8});
            indices(32, 0);
            f.four_reference_symbols = true;
            break;
    }
    return f;
}

// The record's residuals (before the references are added).
PlainBlock plain_residual(const Tables& t, uint32_t bc, const uint8_t* s) {
    PlainBlock r;
    switch (bc) {
        case 1:
            for (uint32_t k = 0; k < 6; ++k) {
                r.ep[k] = unzigzag_mod(s[2 + k], k % 3 == 1 ? 6 : 5);
            }
            quad_indices(t.rank_bc1, s + 8, r.ix);
            break;
        case 3:
            // Colour symbols come per channel (both endpoints together), alpha first.
            for (uint32_t c = 0; c < 3; ++c) {
                const uint32_t bits = c == 1 ? 6 : 5;
                r.ep[c] = unzigzag_mod(s[14 + 2 * c], bits);
                r.ep[3 + c] = unzigzag_mod(s[15 + 2 * c], bits);
            }
            r.ep[6] = unzigzag_mod(s[4], 8);
            r.ep[7] = unzigzag_mod(s[5], 8);
            pair_indices(t.rank_bc3_alpha, s + 6, 8, r.ix);
            quad_indices(t.rank_bc3_color, s + 20, r.ix + 16);
            break;
        case 4:
            r.ep[0] = unzigzag_mod(s[2], 8);
            r.ep[1] = unzigzag_mod(s[3], 8);
            pair_indices(t.rank_bc4, s + 4, 8, r.ix);
            break;
        default:  // 5
            for (uint32_t k = 0; k < 4; ++k) {
                r.ep[k] = unzigzag_mod(s[4 + k], 8);
            }
            pair_indices(t.rank_bc5, s + 8, 16, r.ix);
            break;
    }
    return r;
}

void pack_plain(uint32_t bc, const PlainBlock& v, uint8_t* out) {
    switch (bc) {
        case 1:
            pack_color(v.ep, v.ix, out);
            break;
        case 3:
            pack_alpha(v.ep[6], v.ep[7], v.ix, out);
            pack_color(v.ep, v.ix + 16, out + 8);
            break;
        case 4:
            pack_alpha(v.ep[0], v.ep[1], v.ix, out);
            break;
        default:  // 5
            pack_alpha(v.ep[0], v.ep[1], v.ix, out);
            pack_alpha(v.ep[2], v.ep[3], v.ix + 16, out + 8);
            break;
    }
}

DecodeStatus decode_plain(const Tables& t, uint32_t bc, const uint8_t* records, size_t record_size, size_t blocks,
                          uint8_t* dst) {
    const PlainFormat f = plain_format(bc);
    const std::vector<PlainBlock>& dictionary = t.plain_dictionary[bc];
    const size_t out_bytes = block_bytes(bc);
    std::vector<PlainBlock> history(kHistoryBlocks);
    const PlainBlock zero{};
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t* s = records + i * record_size;
        const References refs = references(s, f.four_reference_symbols);
        const PlainBlock& ep_ref = referenced(i, refs.endpoints, history, dictionary, zero);
        const PlainBlock& ix_ref = referenced(i, refs.indices, history, dictionary, zero);
        PlainBlock v = plain_residual(t, bc, s);
        for (uint32_t k = 0; k < 8; ++k) {
            v.ep[k] = static_cast<uint8_t>((v.ep[k] + ep_ref.ep[k]) & ((1u << f.endpoint_bits[k]) - 1u));
        }
        for (uint32_t k = 0; k < 32; ++k) {
            v.ix[k] = static_cast<uint8_t>((v.ix[k] + ix_ref.ix[k]) & ((1u << f.index_bits[k]) - 1u));
        }
        history[i % kHistoryBlocks] = v;
        pack_plain(bc, v, dst + i * out_bytes);
    }
    return DecodeStatus::Ok;
}

// ---------------------------------------------------------------------------
// BC7
// ---------------------------------------------------------------------------

// BC7 (public format): endpoint bits of R, G, B, A per mode, without p-bits.
constexpr uint8_t kBc7EndpointBits[8][4] = {{4, 4, 4, 0}, {6, 6, 6, 0}, {5, 5, 5, 0}, {7, 7, 7, 0},
                                            {5, 5, 5, 6}, {7, 7, 7, 8}, {7, 7, 7, 7}, {5, 5, 5, 5}};
// Index bits of the primary and secondary index sets per mode.
constexpr uint8_t kBc7IndexBits[8][2] = {{3, 0}, {3, 0}, {2, 0}, {2, 0}, {3, 2}, {2, 2}, {4, 0}, {2, 0}};
constexpr uint8_t kBc7Subsets[8] = {3, 2, 3, 2, 1, 1, 1, 2};
constexpr uint8_t kBc7PartitionBits[8] = {4, 6, 6, 6, 0, 0, 0, 6};

// BC7 anchor ("fix-up") pixels of the second subset (two-subset partitions) and of the second and
// third subsets (three-subset partitions), from the public BC7 specification. Values as published
// in Microsoft's DirectXTex (BC6HBC7.cpp, g_aFixUp; Copyright (c) Microsoft Corporation, MIT
// License).
constexpr uint8_t kBc7Anchor2[64] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                                     15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2,
                                     15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2,  2,  2,  15, 15, 6,
                                     6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15};
constexpr uint8_t kBc7Anchor3[64][2] = {
    {3, 15},  {3, 8},   {15, 8},  {15, 3}, {8, 15},  {3, 15},  {15, 3},  {15, 8},  {8, 15},  {8, 15},  {6, 15},
    {6, 15},  {6, 15},  {5, 15},  {3, 15}, {3, 8},   {3, 15},  {3, 8},   {8, 15},  {15, 3},  {3, 15},  {3, 8},
    {6, 15},  {10, 8},  {5, 3},   {8, 15}, {8, 6},   {6, 10},  {8, 15},  {5, 15},  {15, 10}, {15, 8},  {8, 15},
    {15, 3},  {3, 15},  {5, 10},  {6, 10}, {10, 8},  {8, 9},   {15, 10}, {15, 6},  {3, 15},  {15, 8},  {5, 15},
    {15, 3},  {15, 6},  {15, 6},  {15, 8}, {3, 15},  {15, 3},  {5, 15},  {5, 15},  {5, 15},  {8, 15},  {5, 15},
    {10, 15}, {5, 15},  {10, 15}, {8, 15}, {13, 15}, {15, 3},  {12, 15}, {3, 15},  {3, 8}};

// The BC7 control word (from the BC7C table): bits 0-2 mode, 3-4 rotation, 5 index selection,
// 6-7 subset count, 8-13 partition.
struct Bc7Control {
    uint32_t mode, rotation, index_select, subsets, partition;
};
inline Bc7Control bc7_control(uint16_t c) {
    return {c & 7u, (c >> 3) & 3u, (c >> 5) & 1u, (c >> 6) & 3u, (c >> 8) & 63u};
}

inline bool bc7_has_pbits(uint32_t mode) { return mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7; }

// Stored precision of the R, G, B, A columns: the p-bit is kept as red's lowest bit; a rotation
// swaps alpha with the rotated channel.
std::array<uint32_t, 4> bc7_precision(const Bc7Control& c) {
    std::array<uint32_t, 4> p;
    for (uint32_t k = 0; k < 4; ++k) {
        p[k] = kBc7EndpointBits[c.mode][k];
    }
    if (c.rotation != 0) {
        std::swap(p[3], p[c.rotation - 1]);
    } else if (bc7_has_pbits(c.mode)) {
        p[0] += 1;
    }
    return p;
}

// A 4-bit symbol from alternate bits of `b` (starting at bit `first`), zigzag-decoded to signed.
inline int16_t alternate_bits_signed(uint8_t b, uint32_t first) {
    uint32_t v = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        v |= ((b >> (first + 2 * k)) & 1u) << k;
    }
    return static_cast<int16_t>(static_cast<int32_t>(v >> 1) ^ -static_cast<int32_t>(v & 1u));
}

// Residual plus reference, each left-aligned to 16 bits at its own precision, back to `bits`
// bits (signed).
inline int16_t add_index(int16_t residual, uint32_t bits, int16_t ref, uint32_t ref_bits) {
    auto align = [](int16_t v, uint32_t b) -> uint16_t {
        return (b == 0 || b > 16) ? 0 : static_cast<uint16_t>(static_cast<uint16_t>(v) << (16 - b));
    };
    if (bits == 0) {
        return 0;
    }
    const auto sum = static_cast<int16_t>(static_cast<uint16_t>(align(residual, bits) + align(ref, ref_bits)));
    return static_cast<int16_t>(sum >> (16 - bits));
}

// Little-endian bit writer for one 128-bit block.
class BlockBits {
public:
    void put(uint64_t value, uint32_t n) {
        for (uint32_t k = 0; k < n && at_ < 128; ++k, ++at_) {
            if ((value >> k) & 1u) {
                word_[at_ / 64] |= uint64_t{1} << (at_ % 64);
            }
        }
    }
    void store(uint8_t* out) const {
        write_le(out, word_[0], 8);
        write_le(out + 8, word_[1], 8);
    }

private:
    uint64_t word_[2] = {};
    uint32_t at_ = 0;
};

void pack_bc7(const Bc7Block& b, const std::array<uint32_t, 4>& prec, uint8_t* out) {
    const Bc7Control c = bc7_control(b.control);
    BlockBits w;
    w.put(uint64_t{1} << c.mode, c.mode + 1);
    w.put(c.partition, kBc7PartitionBits[c.mode]);
    if (c.mode == 4 || c.mode == 5) {
        w.put(c.rotation, 2);
    }
    if (c.mode == 4) {
        w.put(c.index_select, 1);
    }
    const uint32_t endpoints = 2 * c.subsets;
    auto field = [&b](uint32_t row, uint32_t column, uint32_t bits) -> uint64_t {
        return b.rows[row][column] >> (16 - bits);
    };
    // Channels in block order R G B A; a rotation swaps where alpha's and the channel's fields go.
    std::array<uint32_t, 4> order = {0, 1, 2, 3};
    if (c.rotation != 0) {
        std::swap(order[3], order[c.rotation - 1]);
    }
    const bool pbits = bc7_has_pbits(c.mode);
    for (uint32_t ch : order) {
        const uint32_t bits = prec[ch];
        if (bits == 0) {
            continue;
        }
        const uint32_t drop = (ch == 0 && pbits) ? 1 : 0;
        for (uint32_t e = 0; e < endpoints; ++e) {
            w.put(field(e / 2, 2 * ch + e % 2, bits) >> drop, bits - drop);
        }
    }
    if (c.mode == 1) {  // one shared p-bit per subset
        for (uint32_t row = 0; row < c.subsets; ++row) {
            w.put(field(row, 0, prec[0]) & 1u, 1);
        }
    } else if (pbits) {
        for (uint32_t e = 0; e < endpoints; ++e) {
            w.put(field(e / 2, e % 2, prec[0]) & 1u, 1);
        }
    }
    auto is_anchor = [&c](uint32_t pixel) {
        if (pixel == 0) {
            return true;
        }
        if (c.subsets == 2) {
            return pixel == kBc7Anchor2[c.partition];
        }
        if (c.subsets == 3) {
            return pixel == kBc7Anchor3[c.partition][0] || pixel == kBc7Anchor3[c.partition][1];
        }
        return false;
    };
    auto put_indices = [&](uint32_t set) {
        const uint32_t bits = b.index_bits[set];
        for (uint32_t p = 0; p < 16; ++p) {
            w.put(static_cast<uint16_t>(b.index[set][p]), bits - (is_anchor(p) ? 1 : 0));
        }
    };
    if (c.mode == 4) {  // the 2-bit set comes first
        put_indices(1);
        put_indices(0);
    } else {
        put_indices(0);
        if (c.mode == 5) {
            put_indices(1);
        }
    }
    w.store(out);
}

DecodeStatus decode_bc7(const Tables& t, const uint8_t* records, size_t blocks, uint8_t* dst) {
    std::vector<Bc7Block> history(kHistoryBlocks);
    const Bc7Block zero{};
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t* s = records + i * 32;
        const References refs = references(s, true);
        const uint16_t control = t.bc7_control[((s[31] & 15u) << 5) | (s[30] & 31u)];
        const Bc7Control c = bc7_control(control);
        if (control == 0 || c.subsets != kBc7Subsets[c.mode]) {
            return DecodeStatus::Corrupt;
        }
        const std::array<uint32_t, 4> prec = bc7_precision(c);
        const bool three = c.subsets == 3;
        // Endpoint residual symbols per subset row (R0 R1 G0 G1 B0 B1 A0 A1). Three-subset blocks
        // have no alpha and use the alpha symbols for the third subset's green and blue.
        const uint8_t stored[3][8] = {
            {s[4], s[5], s[6], s[7], s[8], s[9], three ? uint8_t{0} : s[10], three ? uint8_t{0} : s[11]},
            {s[12], s[13], s[14], s[15], s[16], s[17], three ? uint8_t{0} : s[18], three ? uint8_t{0} : s[19]},
            {s[20], s[21], three ? s[10] : uint8_t{0}, three ? s[11] : uint8_t{0}, three ? s[18] : uint8_t{0},
             three ? s[19] : uint8_t{0}, 0, 0}};
        const Bc7Block& er = referenced(i, refs.endpoints, history, t.bc7_dictionary, zero);
        Bc7Block v;
        v.control = control;
        // Subsets the reference lacks predict from its first subset.
        const uint32_t shared_rows = std::min(c.subsets, bc7_control(er.control).subsets);
        const uint32_t columns = prec[3] > 0 ? 8 : 6;
        for (uint32_t col = 0; col < columns; ++col) {
            const uint32_t bits = prec[col / 2];
            if (bits == 0) {
                continue;
            }
            const auto mask = static_cast<uint16_t>(0xFFFFu << (16 - bits));
            for (uint32_t row = 0; row < c.subsets; ++row) {
                const uint8_t z = stored[row][col];
                const auto residual =
                    static_cast<uint16_t>(static_cast<uint16_t>((z >> 1) ^ (0u - (z & 1u))) << (16 - bits));
                const uint16_t base = er.rows[row < shared_rows ? row : 0][col] & mask;
                v.rows[row][col] = static_cast<uint16_t>(base + residual);
            }
        }
        const Bc7Block& ir = referenced(i, refs.indices, history, t.bc7_dictionary, zero);
        v.index_bits[0] = kBc7IndexBits[c.mode][0];
        v.index_bits[1] = kBc7IndexBits[c.mode][1];
        // Index residuals: four symbol pairs per set, each byte two signed 4-bit values on
        // alternate bits; the first three pairs give the odd bits first.
        constexpr uint32_t kSetSymbols[2][4] = {{22, 24, 26, 28}, {12, 14, 16, 20}};
        for (uint32_t set = 0; set < 2; ++set) {
            const uint32_t bits = v.index_bits[set];
            if (bits == 0) {
                continue;
            }
            for (uint32_t k = 0; k < 4; ++k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t byte = s[kSetSymbols[set][k] + half];
                    const int16_t even = alternate_bits_signed(byte, 0);
                    const int16_t odd = alternate_bits_signed(byte, 1);
                    const int16_t first = k < 3 ? odd : even;
                    const int16_t second = k < 3 ? even : odd;
                    const uint32_t p = 4 * k + 2 * half;
                    v.index[set][p] = add_index(first, bits, ir.index[set][p], ir.index_bits[set]);
                    v.index[set][p + 1] = add_index(second, bits, ir.index[set][p + 1], ir.index_bits[set]);
                }
            }
        }
        history[i % kHistoryBlocks] = v;
        pack_bc7(v, prec, dst + i * 16);
    }
    return DecodeStatus::Ok;
}

struct CursorFetch {
    const uint8_t* p;
    const uint8_t* end;
    bool operator()(size_t, uint8_t* byte) {
        if (p == end) {
            return false;
        }
        *byte = *p++;
        return true;
    }
};

std::mutex g_tables_mutex;
std::shared_ptr<const Tables> g_tables;

}  // namespace

DecodeStatus decode(const Tables& t, uint32_t bc, const uint8_t* src, size_t src_size, uint8_t* dst,
                    size_t dst_size) {
    if (!supports(t, bc)) {
        return DecodeStatus::Unsupported;
    }
    const size_t bytes = block_bytes(bc);
    if (dst == nullptr || dst_size == 0 || dst_size % bytes != 0) {
        return DecodeStatus::BadSize;
    }
    const size_t blocks = dst_size / bytes;
    const size_t record_size = record_symbols(bc);
    std::vector<uint8_t> records(blocks * record_size);
    CursorFetch fetch{src, src + (src != nullptr ? src_size : 0)};
    if (!run_lanes(t, t.layouts[bc], blocks, fetch, records.data())) {
        return DecodeStatus::Truncated;
    }
    if (bc == 7) {
        return decode_bc7(t, records.data(), blocks, dst);
    }
    return decode_plain(t, bc, records.data(), record_size, blocks, dst);
}

bool decode_lanes(const Tables& t, uint32_t bc, size_t blocks, FetchByte fetch, void* context, uint8_t* records) {
    if (bc >= kFormats || !t.layouts[bc].present || fetch == nullptr) {
        return false;
    }
    auto adapter = [fetch, context](size_t lane, uint8_t* byte) { return fetch(context, lane, byte); };
    return run_lanes(t, t.layouts[bc], blocks, adapter, records);
}

bool load_tables_file(const std::string& path, std::string* error) {
    std::string scratch;
    std::string& why = error != nullptr ? *error : scratch;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        why = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> chunk(65536);
    size_t n = 0;
    while (bytes.size() < (64u << 20) && (n = std::fread(chunk.data(), 1, chunk.size(), f)) > 0) {
        bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(n));
    }
    std::fclose(f);
    std::shared_ptr<const Tables> tables = parse_tables(bytes.data(), bytes.size(), &why);
    if (!tables) {
        why = path + ": " + why;
        return false;
    }
    set_active_tables(std::move(tables));
    return true;
}

void set_active_tables(std::shared_ptr<const Tables> tables) {
    std::lock_guard<std::mutex> lock(g_tables_mutex);
    g_tables = std::move(tables);
}

std::shared_ptr<const Tables> active_tables() {
    std::lock_guard<std::mutex> lock(g_tables_mutex);
    return g_tables;
}

}  // namespace playgta6::os::bcpack
