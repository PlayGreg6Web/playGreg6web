#pragma once

// Test-only BCPack helpers: SYNTHETIC table files (made-up models, lane widths, ranks, control
// words and dictionaries; nothing here comes from the console format's real tables) and a tiny
// encoder that turns per-block symbol records into a lane stream the decoder accepts. Used by
// test_bcpack.cpp and the DirectStorage tests in test_xbox_scarlett_guest.cpp.

#include <cstdint>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "os/bcpack.h"

namespace bcpack_synthetic {

namespace bp = playgta6::os::bcpack;

inline void put16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >> 8));
}
inline void put32(std::vector<uint8_t>& v, uint32_t x) {
    put16(v, x & 0xFFFF);
    put16(v, x >> 16);
}

// Synthetic model frequencies: model m (group g = m / 16) uses g + 2 buckets; buckets 1.. get
// small made-up frequencies and bucket 0 (symbol 0) takes the rest of the 65536 slots.
inline std::vector<uint32_t> model_freqs(uint32_t m) {
    const uint32_t used = m / 16 + 2;
    std::vector<uint32_t> f(9, 0);
    uint32_t taken = 0;
    for (uint32_t b = 1; b < used; ++b) {
        f[b] = 1 + ((m * 7 + b * 13) % 50) * 3;
        taken += f[b] << (b - 1);
    }
    f[0] = 65536 - taken;
    return f;
}

struct LaneSpec {
    uint32_t bc;
    uint32_t lanes;
    uint32_t per_block;
    std::vector<uint8_t> widths;
};

// Made-up lane layouts (all widths 7 unless a test asks for others).
inline std::vector<LaneSpec> default_layouts() {
    return {{1, 12, 1, std::vector<uint8_t>(12, 7)},
            {3, 12, 2, std::vector<uint8_t>(12, 7)},
            {4, 12, 1, std::vector<uint8_t>(12, 7)},
            {5, 12, 2, std::vector<uint8_t>(12, 7)},
            {6, 16, 2, std::vector<uint8_t>(16, 7)},
            {7, 16, 2, std::vector<uint8_t>(16, 7)}};
}

struct Builder {
    std::map<std::string, std::vector<uint8_t>> sections;

    // Everything the decoder can use, synthetic.
    static Builder full(const std::vector<LaneSpec>& layouts = default_layouts()) {
        Builder b;
        std::vector<uint8_t>& modl = b.sections["MODL"];
        for (uint32_t m = 0; m < 128; ++m) {
            for (uint32_t f : model_freqs(m)) {
                put16(modl, f);
            }
        }
        b.set_layouts(layouts);
        // Ranks: BC1/BC3-colour symbols are four 2-bit residuals as they are; the 3-bit pair
        // tables split a 6-bit symbol into two 3-bit residuals (low, high).
        std::vector<uint8_t>& rank = b.sections["RANK"];
        for (uint32_t v = 0; v < 256; ++v) {
            rank.push_back(static_cast<uint8_t>(v));
        }
        auto pairs = [&rank] {
            for (uint32_t v = 0; v < 64; ++v) {
                rank.push_back(static_cast<uint8_t>((v & 7) | ((v >> 3) << 4)));
            }
        };
        pairs();  // BC3 alpha
        for (uint32_t v = 0; v < 256; ++v) {
            rank.push_back(static_cast<uint8_t>(v));  // BC3 colour
        }
        pairs();  // BC4
        pairs();  // BC5
        // BC7 control words: code c = (s31 & 15) << 5 | (s30 & 31); a few made-up entries.
        std::vector<uint16_t> ctrl(512, 0);
        ctrl[0] = 6 | (1 << 6);                                // mode 6
        ctrl[1] = 1 | (2 << 6) | (17 << 8);                    // mode 1, partition 17
        ctrl[2] = 5 | (2 << 3) | (1 << 6);                     // mode 5, rotation 2
        ctrl[3] = 0 | (3 << 6) | (0 << 8);                     // mode 0, partition 0
        ctrl[4] = 4 | (1 << 3) | (1 << 5) | (1 << 6);          // mode 4, rotation 1, index select
        ctrl[5] = 6 | (2 << 6);                                // inconsistent: mode 6 with 2 subsets
        std::vector<uint8_t>& c = b.sections["BC7C"];
        for (uint16_t w : ctrl) {
            put16(c, w);
        }
        // Dictionaries: entry k has made-up values derived from k.
        const uint32_t shapes[6][2] = {{0, 0}, {6, 16}, {0, 0}, {8, 32}, {2, 16}, {4, 32}};
        for (uint32_t bc : {1u, 3u, 4u, 5u}) {
            std::vector<uint8_t>& d = b.sections["DIC" + std::to_string(bc)];
            for (uint32_t k = 0; k < 255; ++k) {
                for (uint32_t e = 0; e < shapes[bc][0] + shapes[bc][1]; ++e) {
                    d.push_back(static_cast<uint8_t>((k * 3 + e) & (e < shapes[bc][0] ? 0x1F : 0x03)));
                }
            }
        }
        // BC6H dictionary: entry k has endpoint j, channel c = ((k + 3 j + c) & 63) << 10, index
        // mode 11 (4-bit indices) and index p = (k + p) % 8 - 4.
        std::vector<uint8_t>& d6 = b.sections["DIC6"];
        for (uint32_t k = 0; k < 255; ++k) {
            d6.push_back(0);
            for (uint32_t e = 0; e < 12; ++e) {
                put16(d6, ((k + 3 * (e / 3) + e % 3) & 63) << 10);
            }
            d6.push_back(11);
            for (uint32_t p = 0; p < 16; ++p) {
                put16(d6, static_cast<uint16_t>(static_cast<int>((k + p) % 8) - 4));
            }
        }
        std::vector<uint8_t>& d7 = b.sections["DIC7"];
        for (uint32_t k = 0; k < 255; ++k) {
            put16(d7, 6 | (1 << 6));  // mode 6
            for (uint32_t e = 0; e < 24; ++e) {
                put16(d7, ((k + e) & 0x7F) << 9);
            }
            d7.push_back(4);
            d7.push_back(0);
            for (uint32_t e = 0; e < 32; ++e) {
                put16(d7, e < 16 ? (k + e) % 8 : 0);
            }
        }
        return b;
    }

    void set_layouts(const std::vector<LaneSpec>& layouts) {
        std::vector<uint8_t>& lane = sections["LANE"];
        lane.clear();
        put32(lane, static_cast<uint32_t>(layouts.size()));
        for (const LaneSpec& l : layouts) {
            lane.push_back(static_cast<uint8_t>(l.bc));
            lane.push_back(static_cast<uint8_t>(l.lanes));
            lane.push_back(static_cast<uint8_t>(l.per_block));
            lane.push_back(0);
            for (uint32_t j = 0; j < 16; ++j) {
                lane.push_back(j < l.widths.size() ? l.widths[j] : 0);
            }
        }
    }

    std::vector<uint8_t> file() const {
        std::vector<uint8_t> out = {'P', 'G', '6', 'B', 'C', 'P', 'A', 'K'};
        put32(out, 1);
        put32(out, static_cast<uint32_t>(sections.size()));
        uint32_t offset = 16 + 16 * static_cast<uint32_t>(sections.size());
        for (const auto& [tag, data] : sections) {
            out.insert(out.end(), tag.begin(), tag.end());
            put32(out, 1);
            put32(out, offset);
            put32(out, static_cast<uint32_t>(data.size()));
            offset += static_cast<uint32_t>(data.size());
        }
        for (const auto& [tag, data] : sections) {
            out.insert(out.end(), data.begin(), data.end());
        }
        return out;
    }

    std::shared_ptr<const bp::Tables> tables() const {
        const std::vector<uint8_t> f = file();
        std::string error;
        return bp::parse_tables(f.data(), f.size(), &error);
    }
};

// One rANS fragment holding `symbols` with model (width * 16 + low) of the lane, then a filler
// fragment that parks the lane once its symbols are out (a parked lane holds a byte it never
// consumes and asks for no more).
inline std::vector<uint8_t> lane_bytes(uint32_t width, uint32_t low, const std::vector<uint8_t>& symbols) {
    const uint32_t m = width * 16 + low;
    const std::vector<uint32_t> f = model_freqs(m);
    std::vector<uint32_t> first(9, 0);
    for (uint32_t b = 1; b < 9; ++b) {
        first[b] = first[b - 1] + (f[b - 1] << (b == 1 ? 0 : b - 2));
    }
    uint64_t x = 1u << 24;
    std::vector<uint8_t> emitted;
    for (size_t k = symbols.size(); k-- > 0;) {
        const uint32_t s = symbols[k];
        uint32_t freq;
        uint32_t start;
        if (s == 0) {
            freq = f[0] - 1;
            start = 1;
        } else {
            uint32_t b = 0;
            while ((s >> b) != 0) {
                ++b;  // s lies in [2^(b-1), 2^b)
            }
            freq = f[b];
            start = first[b] + (s - (1u << (b - 1))) * freq;
        }
        while (x >= (static_cast<uint64_t>(freq) << 16)) {
            emitted.push_back(static_cast<uint8_t>(x));
            x >>= 8;
        }
        x = ((x / freq) << 16) + (x % freq) + start;
    }
    for (int k = 0; k < 4; ++k) {
        emitted.push_back(static_cast<uint8_t>(x >> (8 * k)));
    }
    std::vector<uint8_t> payload(emitted.rbegin(), emitted.rend());
    std::vector<uint8_t> out;
    if (payload.size() > 256) {
        const size_t n = payload.size() - 257;
        out = {static_cast<uint8_t>(0x20 | low), static_cast<uint8_t>(n), static_cast<uint8_t>(n >> 8)};
    } else {
        out = {static_cast<uint8_t>(low), static_cast<uint8_t>(payload.size() - 1)};
    }
    out.insert(out.end(), payload.begin(), payload.end());
    const uint8_t filler[] = {0x00, 0x01, 0x00, 0x00};
    out.insert(out.end(), std::begin(filler), std::end(filler));
    return out;
}

// Interleaves the lanes' bytes in the order the decoder's schedule takes them (by running the
// decoder's lane stage against per-lane queues). `records` holds lanes * per_block symbols per
// block. Returns an empty vector if the schedule asked a lane for more than it had.
inline std::vector<uint8_t> stream(const bp::Tables& t, const LaneSpec& spec,
                                   const std::vector<std::vector<uint8_t>>& records) {
    struct Queues {
        std::vector<std::deque<uint8_t>> lanes;
        std::vector<uint8_t> out;
        bool starved = false;
    } q;
    for (uint32_t j = 0; j < spec.lanes; ++j) {
        std::vector<uint8_t> symbols;
        for (const auto& r : records) {
            for (uint32_t k = 0; k < spec.per_block; ++k) {
                symbols.push_back(r[j * spec.per_block + k]);
            }
        }
        const std::vector<uint8_t> bytes = lane_bytes(spec.widths[j], j % 16, symbols);
        q.lanes.emplace_back(bytes.begin(), bytes.end());
    }
    auto fetch = [](void* ctx, size_t lane, uint8_t* byte) {
        auto* qs = static_cast<Queues*>(ctx);
        if (qs->lanes[lane].empty()) {
            qs->starved = true;
            return false;
        }
        *byte = qs->lanes[lane].front();
        qs->lanes[lane].pop_front();
        qs->out.push_back(*byte);
        return true;
    };
    std::vector<uint8_t> check(records.size() * spec.lanes * spec.per_block);
    if (!bp::decode_lanes(t, spec.bc, records.size(), fetch, &q, check.data()) || q.starved) {
        return {};
    }
    return q.out;
}

inline uint8_t zigzag(int v) { return static_cast<uint8_t>(v >= 0 ? 2 * v : -2 * v - 1); }

}  // namespace bcpack_synthetic
