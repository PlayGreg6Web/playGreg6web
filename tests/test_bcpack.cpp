// BCPack decoder (os/bcpack.h) on SYNTHETIC data only: made-up table files (bcpack_synthetic.h),
// streams from a test-only encoder, and hand-built block records. No real tables, no game data.
// Decoded BC blocks are checked against the public BC1/BC4/BC7 block layouts, read back here.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "bcpack_synthetic.h"
#include "os/bcpack.h"

namespace {

int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_bcpack.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace bp = playgta6::os::bcpack;
namespace sy = bcpack_synthetic;
using Record = std::vector<uint8_t>;

const sy::LaneSpec& spec_for(uint32_t bc) {
    static const std::vector<sy::LaneSpec> specs = sy::default_layouts();
    for (const sy::LaneSpec& s : specs) {
        if (s.bc == bc) {
            return s;
        }
    }
    return specs[0];
}

std::vector<uint8_t> decode_records(const bp::Tables& t, uint32_t bc, const std::vector<Record>& records,
                                    bp::DecodeStatus* status = nullptr) {
    const std::vector<uint8_t> src = sy::stream(t, spec_for(bc), records);
    CHECK(!src.empty());
    std::vector<uint8_t> dst(records.size() * bp::block_bytes(bc));
    const bp::DecodeStatus st = bp::decode(t, bc, src.data(), src.size(), dst.data(), dst.size());
    if (status != nullptr) {
        *status = st;
    } else {
        CHECK(st == bp::DecodeStatus::Ok);
    }
    return dst;
}

// Signed equivalent of `v` modulo 2^bits, zigzagged (what a residual symbol holds).
uint8_t residual(uint32_t v, uint32_t bits) {
    const int half = 1 << (bits - 1);
    const int s = static_cast<int>(v & ((1u << bits) - 1)) >= half ? static_cast<int>(v & ((1u << bits) - 1)) - (1 << bits)
                                                                    : static_cast<int>(v & ((1u << bits) - 1));
    return sy::zigzag(s);
}

uint64_t le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int k = 7; k >= 0; --k) {
        v = (v << 8) | p[k];
    }
    return v;
}

// Public BC1 (four-colour mode) / BC4 index codes -> position along the endpoint line.
uint32_t bc1_position(uint32_t code) {
    const uint32_t pos[4] = {0, 3, 1, 2};
    return pos[code];
}
uint32_t bc4_position(uint32_t code) { return code == 0 ? 0 : code == 1 ? 7 : code - 1; }

// ---------------------------------------------------------------------------
// Table files
// ---------------------------------------------------------------------------

void test_table_parsing() {
    const sy::Builder full = sy::Builder::full();
    std::string error;
    const std::vector<uint8_t> file = full.file();
    const auto t = bp::parse_tables(file.data(), file.size(), &error);
    CHECK(t != nullptr);
    if (!t) {
        std::fprintf(stderr, "  parse error: %s\n", error.c_str());
        return;
    }
    for (uint32_t bc : {1u, 3u, 4u, 5u, 6u, 7u}) {
        CHECK(bp::supports(*t, bc));
    }
    CHECK(!bp::supports(*t, 2) && !bp::supports(*t, 0) && !bp::supports(*t, 15));
    CHECK(bp::supported_formats(*t) == "BC1, BC3, BC4, BC5, BC6H, BC7");

    auto parses = [](const sy::Builder& b, std::string* why = nullptr) {
        const std::vector<uint8_t> f = b.file();
        std::string e;
        const bool ok = bp::parse_tables(f.data(), f.size(), &e) != nullptr;
        if (why != nullptr) {
            *why = e;
        }
        return ok;
    };
    // Optional sections: without DIC7 BC7 is off, the rest still works; unknown tags are ignored.
    sy::Builder no_dic7 = full;
    no_dic7.sections.erase("DIC7");
    no_dic7.sections["ZZZZ"] = {1, 2, 3};
    {
        const std::vector<uint8_t> f = no_dic7.file();
        const auto t2 = bp::parse_tables(f.data(), f.size(), nullptr);
        CHECK(t2 != nullptr && !bp::supports(*t2, 7) && bp::supports(*t2, 4));
        CHECK(t2 && bp::supported_formats(*t2) == "BC1, BC3, BC4, BC5, BC6H");
    }
    sy::Builder bad_dic6 = full;
    bad_dic6.sections["DIC6"].pop_back();
    CHECK(!parses(bad_dic6, &error) && error.find("DIC6") != std::string::npos);
    bad_dic6 = full;
    bad_dic6.sections["DIC6"][25] = 15;  // entry 0's index-half mode
    CHECK(!parses(bad_dic6, &error) && error.find("mode 15") != std::string::npos);
    // Required sections, sizes, model tiling, lane widths.
    sy::Builder no_lane = full;
    no_lane.sections.erase("LANE");
    CHECK(!parses(no_lane, &error) && error.find("LANE") != std::string::npos);
    sy::Builder bad_model = full;
    bad_model.sections["MODL"][0] ^= 1;  // model 0 no longer covers 65536 slots
    CHECK(!parses(bad_model, &error) && error.find("model 0") != std::string::npos);
    sy::Builder empty_bucket = full;
    empty_bucket.sections["MODL"][2] = empty_bucket.sections["MODL"][3] = 0;
    CHECK(!parses(empty_bucket, &error) && error.find("empty bucket") != std::string::npos);
    sy::Builder short_rank = full;
    short_rank.sections["RANK"].pop_back();
    CHECK(!parses(short_rank));
    sy::Builder narrow = full;
    std::vector<sy::LaneSpec> layouts = sy::default_layouts();
    layouts[2].widths[5] = 2;  // below 3: the header could select a negative group
    narrow.set_layouts(layouts);
    CHECK(!parses(narrow, &error) && error.find("width") != std::string::npos);
    sy::Builder wrong_record = full;
    layouts = sy::default_layouts();
    layouts[4].lanes = 12;  // 12 x 2 symbols is not BC7's 32-symbol record
    wrong_record.set_layouts(layouts);
    CHECK(!parses(wrong_record));
    // Header problems.
    std::vector<uint8_t> f = file;
    f[0] = 'X';
    CHECK(bp::parse_tables(f.data(), f.size(), &error) == nullptr && error.find("magic") != std::string::npos);
    f = file;
    f[8] = 2;  // file version
    CHECK(bp::parse_tables(f.data(), f.size(), nullptr) == nullptr);
    f = file;
    f.resize(f.size() - 1);  // the last section runs past the end
    CHECK(bp::parse_tables(f.data(), f.size(), nullptr) == nullptr);
    CHECK(bp::parse_tables(nullptr, 0, nullptr) == nullptr);
    CHECK(bp::parse_tables(file.data(), 15, nullptr) == nullptr);
}

void test_tables_file_and_active() {
    bp::set_active_tables(nullptr);
    CHECK(bp::active_tables() == nullptr);
    std::string error;
    CHECK(!bp::load_tables_file("this/file/does/not/exist.bin", &error));
    CHECK(error.find("cannot open") != std::string::npos);
    const std::vector<uint8_t> file = sy::Builder::full().file();
    const std::string path = "test_bcpack_synthetic_tables.tmp";
    std::FILE* out = std::fopen(path.c_str(), "wb");
    CHECK(out != nullptr);
    if (out != nullptr) {
        std::fwrite(file.data(), 1, file.size(), out);
        std::fclose(out);
        CHECK(bp::load_tables_file(path, &error));
        CHECK(bp::active_tables() != nullptr && bp::supports(*bp::active_tables(), 7));
        std::remove(path.c_str());
    }
    bp::set_active_tables(nullptr);
    CHECK(bp::active_tables() == nullptr);
}

// ---------------------------------------------------------------------------
// Lanes: the rANS fragments and the byte schedule round-trip
// ---------------------------------------------------------------------------

void test_lane_round_trip() {
    // Mixed made-up widths: symbols stay below 2^(width + 1).
    std::vector<sy::LaneSpec> layouts = {
        {1, 12, 1, {7, 6, 5, 4, 3, 7, 6, 5, 4, 3, 7, 7}},
        {3, 12, 2, {3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 7}},
        {4, 12, 1, std::vector<uint8_t>(12, 7)},
        {5, 12, 2, {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}},
        {7, 16, 2, {3, 3, 7, 7, 6, 6, 5, 5, 4, 4, 7, 7, 6, 5, 4, 3}},
    };
    const auto t = sy::Builder::full(layouts).tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    uint64_t seed = 0x243F6A8885A308D3ull;
    auto next = [&seed] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (const sy::LaneSpec& spec : layouts) {
        for (size_t blocks : {size_t{1}, size_t{2}, size_t{17}, size_t{300}, size_t{1500}}) {
            std::vector<Record> records(blocks, Record(spec.lanes * spec.per_block));
            for (Record& r : records) {
                for (size_t k = 0; k < r.size(); ++k) {
                    const uint32_t alphabet = 1u << (spec.widths[k / spec.per_block] + 1);
                    // Mostly small symbols (cheap), some anywhere in the alphabet.
                    const uint64_t v = next();
                    r[k] = static_cast<uint8_t>((v & 3) == 0 ? v % alphabet : (v >> 8) % 4);
                }
            }
            const std::vector<uint8_t> src = sy::stream(*t, spec, records);
            CHECK(!src.empty());
            struct Cursor {
                const std::vector<uint8_t>* src;
                size_t at;
            } cursor{&src, 0};
            auto fetch = [](void* ctx, size_t, uint8_t* byte) {
                auto* c = static_cast<Cursor*>(ctx);
                if (c->at >= c->src->size()) {
                    return false;
                }
                *byte = (*c->src)[c->at++];
                return true;
            };
            std::vector<uint8_t> got(blocks * spec.lanes * spec.per_block);
            CHECK(bp::decode_lanes(*t, spec.bc, blocks, fetch, &cursor, got.data()));
            std::vector<uint8_t> want;
            for (const Record& r : records) {
                want.insert(want.end(), r.begin(), r.end());
            }
            CHECK(got == want);
            if (blocks == 1500) {
                CHECK(src.size() > 12 * 260);  // long (two-byte length) fragments were exercised
            }
            // Cut short: the lane stage reports it instead of looping or reading past the end.
            const std::vector<uint8_t> half(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(src.size() / 2));
            Cursor short_cursor{&half, 0};
            CHECK(!bp::decode_lanes(*t, spec.bc, blocks, fetch, &short_cursor, got.data()));
        }
    }
}

// ---------------------------------------------------------------------------
// Plain formats
// ---------------------------------------------------------------------------

// BC1 symbol for one 2x2 quad: the four 2-bit residuals of pixels (top-left, top-right,
// bottom-left, bottom-right) with the synthetic identity rank table.
uint8_t quad_symbol(const uint32_t ix[16], uint32_t q) {
    const uint32_t corner = (q / 2) * 8 + (q % 2) * 2;
    const uint32_t px[4] = {corner, corner + 1, corner + 4, corner + 5};
    uint32_t v = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        v |= (ix[px[k]] & 3u) << (2 * k);
    }
    return static_cast<uint8_t>(v);
}

// Checks a BC1 colour half against endpoints and line positions.
void check_color_half(const uint8_t* half, const uint32_t c0[3], const uint32_t c1[3], const uint32_t pos[16]) {
    const uint64_t v = le64(half);
    CHECK((v & 0xFFFF) == ((c0[0] << 11) | (c0[1] << 5) | c0[2]));
    CHECK(((v >> 16) & 0xFFFF) == ((c1[0] << 11) | (c1[1] << 5) | c1[2]));
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(bc1_position((v >> (32 + 2 * p)) & 3) == pos[p]);
    }
}

// Checks a BC4 half: endpoints are stored with the top bit flipped.
void check_alpha_half(const uint8_t* half, uint32_t e0, uint32_t e1, const uint32_t pos[16]) {
    const uint64_t v = le64(half);
    CHECK((v & 0xFF) == (e0 ^ 0x80));
    CHECK(((v >> 8) & 0xFF) == (e1 ^ 0x80));
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(bc4_position((v >> (16 + 3 * p)) & 7) == pos[p]);
    }
}

void test_bc1_references() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    const uint32_t c0[3] = {31, 40, 7};
    const uint32_t c1[3] = {2, 10, 30};
    uint32_t pos0[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos0[p] = (p * 3 + 1) % 4;
    }
    // Block 0: no references.
    Record b0(12, 0);
    for (uint32_t k = 0; k < 3; ++k) {
        b0[2 + k] = residual(c0[k], k == 1 ? 6 : 5);
        b0[5 + k] = residual(c1[k], k == 1 ? 6 : 5);
    }
    for (uint32_t q = 0; q < 4; ++q) {
        b0[8 + q] = quad_symbol(pos0, q);
    }
    // Block 1: both references to block 0 (index reference 1, endpoint delta 0); residuals +1.
    Record b1(12, 0);
    b1[1] = 1;
    for (uint32_t k = 0; k < 6; ++k) {
        b1[2 + k] = sy::zigzag(1);
    }
    const uint32_t ones[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (uint32_t q = 0; q < 4; ++q) {
        b1[8 + q] = quad_symbol(ones, q);
    }
    // Block 2: indices from dictionary entry 0 (reference 3 > block 2), endpoints from block 1
    // (delta -2 from the index reference); no residuals.
    Record b2(12, 0);
    b2[0] = sy::zigzag(-2);
    b2[1] = 3;
    const std::vector<uint8_t> dst = decode_records(*t, 1, {b0, b1, b2});
    check_color_half(dst.data(), c0, c1, pos0);
    const uint32_t c0b[3] = {0, 41, 8};  // 31 + 1 wraps at 5 bits
    const uint32_t c1b[3] = {3, 11, 31};
    uint32_t pos1[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos1[p] = (pos0[p] + 1) % 4;
    }
    check_color_half(dst.data() + 8, c0b, c1b, pos1);
    uint32_t dict_pos[16];
    for (uint32_t p = 0; p < 16; ++p) {
        dict_pos[p] = (p + 6) & 3;  // synthetic DIC1 entry 0: index e (6..21) holds e & 3
    }
    check_color_half(dst.data() + 16, c0b, c1b, dict_pos);
}

void test_bc4_bc5_bc3() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    uint32_t pos[16];
    uint32_t pos_next[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos[p] = p & 7;
        pos_next[p] = (p + 1) & 7;
    }
    auto pair_symbols = [](const uint32_t* ix, uint8_t* out, size_t pairs) {
        for (size_t k = 0; k < pairs; ++k) {
            out[k] = static_cast<uint8_t>((ix[2 * k] & 7) | ((ix[2 * k + 1] & 7) << 3));
        }
    };
    // BC4: endpoints 10 and -3 (253), then a block predicting from it with +1 everywhere.
    Record b0(12, 0);
    b0[2] = sy::zigzag(10);
    b0[3] = sy::zigzag(-3);
    pair_symbols(pos, &b0[4], 8);
    Record b1(12, 0);
    b1[1] = 1;
    b1[2] = b1[3] = sy::zigzag(1);
    for (size_t k = 0; k < 8; ++k) {
        b1[4 + k] = 1 | (1 << 3);
    }
    std::vector<uint8_t> dst = decode_records(*t, 4, {b0, b1});
    check_alpha_half(dst.data(), 10, 253, pos);
    check_alpha_half(dst.data() + 8, 11, 254, pos_next);

    // BC5: two BC4 halves; references use four symbols (two nibble-interleaved bytes).
    Record c(24, 0);
    for (uint32_t k = 0; k < 4; ++k) {
        c[4 + k] = residual(40 * k + 5, 8);
    }
    uint32_t pos32[32];
    for (uint32_t p = 0; p < 32; ++p) {
        pos32[p] = (p * 5) & 7;
    }
    pair_symbols(pos32, &c[8], 16);
    dst = decode_records(*t, 5, {c});
    check_alpha_half(dst.data(), 5, 45, pos32);
    check_alpha_half(dst.data() + 8, 85, 125, pos32 + 16);

    // BC3: alpha (symbols 4-13) in the first half, colour (14-23) in the second.
    Record d(24, 0);
    d[4] = residual(200, 8);
    d[5] = residual(17, 8);
    pair_symbols(pos, &d[6], 8);
    const uint32_t c0[3] = {20, 50, 3};
    const uint32_t c1[3] = {9, 1, 28};
    for (uint32_t ch = 0; ch < 3; ++ch) {
        d[14 + 2 * ch] = residual(c0[ch], ch == 1 ? 6 : 5);
        d[15 + 2 * ch] = residual(c1[ch], ch == 1 ? 6 : 5);
    }
    uint32_t cpos[16];
    for (uint32_t p = 0; p < 16; ++p) {
        cpos[p] = (p / 4) & 3;
    }
    for (uint32_t q = 0; q < 4; ++q) {
        d[20 + q] = quad_symbol(cpos, q);
    }
    dst = decode_records(*t, 3, {d});
    check_alpha_half(dst.data(), 200, 17, pos);
    check_color_half(dst.data() + 8, c0, c1, cpos);

    // BC3 with both references into the dictionary via the nibble-interleaved symbols: index
    // symbols (2, 0) merge to 4 (even nibble bit 1 -> bit 2), endpoint delta 0. Reference 4 of
    // block 0 is dictionary entry 3. Synthetic DIC3 entry 3: endpoint e (0..7) = 9 + e, index j
    // (0..31) = (17 + j) & 3.
    Record e(24, 0);
    e[2] = 2;
    dst = decode_records(*t, 3, {e});
    uint32_t ea[8];
    for (uint32_t k = 0; k < 8; ++k) {
        ea[k] = (9 + k) & 0x1F;
    }
    const uint32_t ec0[3] = {ea[0], ea[1], ea[2]};
    const uint32_t ec1[3] = {ea[3], ea[4], ea[5]};
    uint32_t apos[16];
    uint32_t cpos2[16];
    for (uint32_t p = 0; p < 16; ++p) {
        apos[p] = ((9 + 8 + p) & 3) & 7;
        cpos2[p] = (9 + 24 + p) & 3;
    }
    check_alpha_half(dst.data(), ea[6], ea[7], apos);
    check_color_half(dst.data() + 8, ec0, ec1, cpos2);
}

// ---------------------------------------------------------------------------
// BC7
// ---------------------------------------------------------------------------

class BitReader {
public:
    explicit BitReader(const uint8_t* b) : b_(b) {}
    uint32_t get(uint32_t n) {
        uint32_t v = 0;
        for (uint32_t k = 0; k < n; ++k, ++at_) {
            v |= static_cast<uint32_t>((b_[at_ / 8] >> (at_ % 8)) & 1u) << k;
        }
        return v;
    }
    uint32_t at() const { return at_; }

private:
    const uint8_t* b_;
    uint32_t at_ = 0;
};

// A BC7 block read with the public layout.
struct Bc7Fields {
    uint32_t mode = 99, partition = 0, rotation = 0, index_select = 0;
    uint32_t ep[3][2][4] = {};  // subset, endpoint, R G B A (field values, without p-bits)
    uint32_t pbit[3][2] = {};   // mode 1: pbit[s][0] is the subset's shared bit
    uint32_t index[2][16] = {};  // in block order: the first index block, then the second
    uint32_t bits = 0;
};

Bc7Fields read_bc7(const uint8_t* block, std::initializer_list<uint32_t> anchors) {
    static const uint32_t kColor[8] = {4, 6, 5, 7, 5, 7, 7, 5};
    static const uint32_t kAlpha[8] = {0, 0, 0, 0, 6, 8, 7, 5};
    static const uint32_t kSubsets[8] = {3, 2, 3, 2, 1, 1, 1, 2};
    static const uint32_t kPartBits[8] = {4, 6, 6, 6, 0, 0, 0, 6};
    Bc7Fields f;
    BitReader r(block);
    uint32_t mode = 0;
    while (mode < 8 && r.get(1) == 0) {
        ++mode;
    }
    f.mode = mode;
    if (mode >= 8) {
        return f;
    }
    f.partition = r.get(kPartBits[mode]);
    if (mode == 4 || mode == 5) {
        f.rotation = r.get(2);
    }
    if (mode == 4) {
        f.index_select = r.get(1);
    }
    const uint32_t subsets = kSubsets[mode];
    for (uint32_t ch = 0; ch < 4; ++ch) {
        const uint32_t n = ch < 3 ? kColor[mode] : kAlpha[mode];
        for (uint32_t s = 0; s < subsets; ++s) {
            for (uint32_t e = 0; e < 2; ++e) {
                f.ep[s][e][ch] = r.get(n);
            }
        }
    }
    if (mode == 1) {
        for (uint32_t s = 0; s < subsets; ++s) {
            f.pbit[s][0] = r.get(1);
        }
    } else if (mode == 0 || mode == 3 || mode == 6 || mode == 7) {
        for (uint32_t s = 0; s < subsets; ++s) {
            for (uint32_t e = 0; e < 2; ++e) {
                f.pbit[s][e] = r.get(1);
            }
        }
    }
    static const uint32_t kFirstIndex[8] = {3, 3, 2, 2, 2, 2, 4, 2};
    static const uint32_t kSecondIndex[8] = {0, 0, 0, 0, 3, 2, 0, 0};
    auto read_set = [&](uint32_t set, uint32_t bits) {
        for (uint32_t p = 0; p < 16; ++p) {
            bool anchor = false;
            for (uint32_t a : anchors) {
                anchor = anchor || a == p;
            }
            f.index[set][p] = r.get(bits - (anchor ? 1 : 0));
        }
    };
    read_set(0, kFirstIndex[mode]);
    if (kSecondIndex[mode] != 0) {
        read_set(1, kSecondIndex[mode]);
    }
    f.bits = r.at();
    return f;
}

// A BC7 record for control code `code` with endpoint column values (R0 R1 G0 G1 B0 B1 A0 A1 per
// subset, at the column's stored precision) and index values of both sets.
Record bc7_record(uint32_t code, const uint32_t ep[3][8], const uint32_t prec[4], const uint32_t ix0[16],
                  const uint32_t ix1[16], bool three) {
    Record s(32, 0);
    s[30] = static_cast<uint8_t>(code & 31);
    s[31] = static_cast<uint8_t>(code >> 5);
    auto put = [&](size_t at, uint32_t row, uint32_t col) { s[at] = residual(ep[row][col], prec[col / 2]); };
    for (uint32_t col = 0; col < 6; ++col) {
        put(4 + col, 0, col);
        put(12 + col, 1, col);
    }
    if (three) {
        put(20, 2, 0);
        put(21, 2, 1);
        put(10, 2, 2);
        put(11, 2, 3);
        put(18, 2, 4);
        put(19, 2, 5);
    } else if (prec[3] != 0) {
        put(10, 0, 6);
        put(11, 0, 7);
        put(18, 1, 6);
        put(19, 1, 7);
    }
    auto nib = [](uint32_t v) {  // a 4-bit signed residual holding v modulo 16
        return static_cast<uint32_t>(sy::zigzag(v >= 8 ? static_cast<int>(v) - 16 : static_cast<int>(v)));
    };
    auto spread = [](uint32_t even, uint32_t odd) {
        uint32_t b = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            b |= ((even >> k) & 1u) << (2 * k);
            b |= ((odd >> k) & 1u) << (2 * k + 1);
        }
        return static_cast<uint8_t>(b);
    };
    const uint32_t offsets[2][4] = {{22, 24, 26, 28}, {12, 14, 16, 20}};
    for (uint32_t set = 0; set < 2; ++set) {
        const uint32_t* ix = set == 0 ? ix0 : ix1;
        if (ix == nullptr) {
            continue;
        }
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t p = 4 * k + 2 * half;
                const uint32_t first = nib(ix[p]);
                const uint32_t second = nib(ix[p + 1]);
                s[offsets[set][k] + half] = k < 3 ? spread(second, first) : spread(first, second);
            }
        }
    }
    return s;
}

void test_bc7_modes() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Mode 6 (code 0): R keeps its p-bit as the lowest of 8 bits; G B A 7 bits.
    {
        const uint32_t prec[4] = {8, 7, 7, 7};
        const uint32_t ep[3][8] = {{0x55, 0xFF, 3, 0, 100, 123, 63, 64}, {}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = p == 0 ? 5 : (p * 7) % 16;  // anchor 0 below 8
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(0, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 6 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 0x2A && f.ep[0][1][0] == 0x7F);
        CHECK(f.pbit[0][0] == 1 && f.pbit[0][1] == 1);
        CHECK(f.ep[0][0][1] == 3 && f.ep[0][1][1] == 0 && f.ep[0][0][2] == 100 && f.ep[0][1][2] == 123);
        CHECK(f.ep[0][0][3] == 63 && f.ep[0][1][3] == 64);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 1 (code 1, partition 17: second-subset anchor at pixel 2): one p-bit per subset.
    {
        const uint32_t prec[4] = {7, 6, 6, 0};
        const uint32_t ep[3][8] = {{0x41, 0x10, 1, 2, 3, 4, 0, 0}, {0x22, 0x7F, 60, 61, 62, 63, 0, 0}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 2) ? p / 2 + 1 : (p * 3 + 2) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(1, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 2});
        CHECK(f.mode == 1 && f.partition == 17 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 0x20 && f.ep[0][1][0] == 0x08 && f.pbit[0][0] == 1);
        CHECK(f.ep[1][0][0] == 0x11 && f.ep[1][1][0] == 0x3F && f.pbit[1][0] == 0);
        CHECK(f.ep[0][0][1] == 1 && f.ep[0][1][2] == 4 && f.ep[1][0][1] == 60 && f.ep[1][1][2] == 63);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 0 (code 3, partition 0: anchors 0, 3, 15): three subsets, the third's G and B come from
    // the symbols alpha uses elsewhere.
    {
        const uint32_t prec[4] = {5, 4, 4, 0};
        const uint32_t ep[3][8] = {
            {1, 2, 3, 4, 5, 6, 0, 0}, {7, 8, 9, 10, 11, 12, 0, 0}, {31, 30, 15, 14, 13, 1, 0, 0}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 3 || p == 15) ? 3 : (p + 4) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(3, ep, prec, ix, nullptr, true)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 3, 15});
        CHECK(f.mode == 0 && f.partition == 0 && f.bits == 128);
        CHECK(f.ep[2][0][0] == 15 && f.pbit[2][0] == 1 && f.ep[2][1][0] == 15 && f.pbit[2][1] == 0);
        CHECK(f.ep[2][0][1] == 15 && f.ep[2][1][1] == 14 && f.ep[2][0][2] == 13 && f.ep[2][1][2] == 1);
        CHECK(f.ep[1][0][0] == 3 && f.pbit[1][0] == 1 && f.ep[1][1][2] == 12);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 5 (code 2, rotation 2): green and alpha trade places, so the 8-bit column pair is the
    // one written in alpha's place.
    {
        const uint32_t prec[4] = {7, 8, 7, 7};
        const uint32_t ep[3][8] = {{10, 20, 200, 201, 30, 40, 50, 60}, {}, {}};
        uint32_t ix0[16];
        uint32_t ix1[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix0[p] = p == 0 ? 1 : p % 4;
            ix1[p] = p == 0 ? 0 : (p + 1) % 4;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(2, ep, prec, ix0, ix1, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 5 && f.rotation == 2 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 10 && f.ep[0][1][0] == 20);
        CHECK(f.ep[0][0][1] == 50 && f.ep[0][1][1] == 60);    // the A columns in G's place
        CHECK(f.ep[0][0][3] == 200 && f.ep[0][1][3] == 201);  // the G columns in A's place
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix0[p] && f.index[1][p] == ix1[p]);
        }
    }
    // Mode 4 (code 4, rotation 1, index selection 1): the 2-bit index set is written first.
    {
        const uint32_t prec[4] = {6, 5, 5, 5};
        const uint32_t ep[3][8] = {{40, 41, 1, 2, 3, 4, 30, 31}, {}, {}};
        uint32_t ix0[16];
        uint32_t ix1[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix0[p] = p == 0 ? 3 : (p * 5) % 8;
            ix1[p] = p == 0 ? 1 : (p * 3) % 4;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(4, ep, prec, ix0, ix1, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 4 && f.rotation == 1 && f.index_select == 1 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 30 && f.ep[0][1][0] == 31);  // A columns in R's place
        CHECK(f.ep[0][0][3] == 40 && f.ep[0][1][3] == 41);  // R columns (6 bits) in A's place
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix1[p]);  // first in the block: the 2-bit set
            CHECK(f.index[1][p] == ix0[p]);
        }
    }
}

void test_bc7_references() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Block 0: mode 6 against dictionary entry 0 for both (references 1). Synthetic DIC7 entry 0:
    // rows[r][c] = ((r * 8 + c) & 0x7F) << 9, index bits 4, primary index p = p % 8.
    const uint32_t prec6[4] = {8, 7, 7, 7};
    const uint32_t zero_ep[3][8] = {};
    const uint32_t zero_ix[16] = {};
    Record b0 = bc7_record(0, zero_ep, prec6, zero_ix, nullptr, false);
    b0[2] = 1;  // index reference: merged nibbles (1, 0) = 1; endpoint delta 0 -> 1
    // Block 1: mode 1 (two subsets) with endpoints from block 0: its second subset predicts from
    // block 0's first; residual +1 on G0 of both subsets; indices fresh.
    const uint32_t prec1[4] = {7, 6, 6, 0};
    uint32_t ep1[3][8] = {};
    ep1[0][2] = ep1[1][2] = 1;
    uint32_t ix1[16];
    for (uint32_t p = 0; p < 16; ++p) {
        ix1[p] = (p == 0 || p == 2) ? 0 : p % 8;
    }
    Record b1 = bc7_record(1, ep1, prec1, ix1, nullptr, false);
    b1[1] = 1;  // odd nibble bit 0: merged delta symbol 2 = +1 -> endpoint reference 1 (block 0)
    const std::vector<uint8_t> dst = decode_records(*t, 7, {b0, b1});
    const Bc7Fields f0 = read_bc7(dst.data(), {0});
    CHECK(f0.mode == 6);
    // Dictionary row 0: column c holds c << 9 as a 16-bit left-aligned value: R (8 bits) = c << 1.
    CHECK(f0.ep[0][0][0] == 0 && f0.ep[0][1][0] == 1 && f0.pbit[0][1] == 0);  // R1 = 2: field 1, p 0
    CHECK(f0.ep[0][0][1] == 2 && f0.ep[0][1][1] == 3);                         // G (7 bits) = c
    for (uint32_t p = 0; p < 16; ++p) {
        // Dictionary index p % 8 at 4 bits, re-read at 4 bits (anchor 0 holds 0).
        CHECK(f0.index[0][p] == p % 8);
    }
    const Bc7Fields f1 = read_bc7(dst.data() + 16, {0, 2});
    CHECK(f1.mode == 1 && f1.partition == 17);
    // Block 0's R0 = 0 at 8 bits -> mode 1 R (7 bits incl. p-bit) 0; G0 = 2 (7 bits) cut to 6
    // bits = 1, + 1 = 2; both subsets.
    CHECK(f1.ep[0][0][1] == 2 && f1.ep[1][0][1] == 2);
    CHECK(f1.ep[0][1][1] == 1 && f1.ep[1][1][1] == 1);  // G1 = 3 cut to 6 bits
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(f1.index[0][p] == ix1[p]);
    }
}

// ---------------------------------------------------------------------------
// BC6H
// ---------------------------------------------------------------------------

// Reads `n` bits at bit `at` of a 128-bit block (little-endian bit order).
uint32_t block_bits(const uint8_t* blk, uint32_t at, uint32_t n) {
    uint32_t v = 0;
    for (uint32_t k = 0; k < n; ++k) {
        v |= static_cast<uint32_t>((blk[(at + k) / 8] >> ((at + k) % 8)) & 1u) << k;
    }
    return v;
}

// A block record for a BC6H block with no references: endpoint fields f[endpoint w x y z][R G B]
// at the mode's precision, stored indices (0 .. 2^bits-1) and the partition (two-subset modes).
Record bc6_record(uint32_t mode, const uint32_t f[4][3], const uint32_t* idx, uint32_t partition) {
    static const uint8_t bits[15][4] = {{0, 0, 0, 0},  {10, 5, 5, 5}, {7, 6, 6, 6},   {11, 5, 4, 4},
                                        {11, 4, 5, 4}, {11, 4, 4, 5}, {9, 5, 5, 5},   {8, 6, 5, 5},
                                        {8, 5, 6, 5},  {8, 5, 5, 6},  {6, 6, 6, 6},   {10, 10, 10, 10},
                                        {11, 9, 9, 9}, {12, 8, 8, 8}, {16, 4, 4, 4}};
    const bool two = mode < 11;
    // Residual symbol: the working value (field, top bit flipped, left-aligned) taken as a signed
    // 16-bit number, zigzagged.
    auto symbol = [](uint32_t field, uint32_t sh) {
        const uint32_t ep = ((field << sh) ^ 0x8000u) & 0xFFFFu;
        const int32_t s = static_cast<int16_t>(ep >> sh);
        return static_cast<uint32_t>(s >= 0 ? 2 * s : -2 * s - 1) & 0xFFFFu;
    };
    auto split = [](uint32_t v, uint8_t* even, uint8_t* odd) {  // the inverse of merging two bytes
        uint32_t a = 0;
        uint32_t b = 0;
        for (uint32_t k = 0; k < 8; ++k) {
            a |= ((v >> (2 * k)) & 1u) << k;
            b |= ((v >> (2 * k + 1)) & 1u) << k;
        }
        *even = static_cast<uint8_t>(a);
        *odd = static_cast<uint8_t>(b);
    };
    Record r(32, 0);
    for (uint32_t c = 0; c < 3; ++c) {
        split(symbol(f[0][c], 16 - bits[mode][0]), &r[4 + 2 * c], &r[5 + 2 * c]);
        split(symbol(f[1][c], 16 - bits[mode][1 + c]), &r[10 + 2 * c], &r[11 + 2 * c]);
        if (two) {
            r[16 + 2 * c] = static_cast<uint8_t>(symbol(f[2][c], 16 - bits[mode][1 + c]));
            r[17 + 2 * c] = static_cast<uint8_t>(symbol(f[3][c], 16 - bits[mode][1 + c]));
        }
    }
    const int ib = mode < 11 ? 3 : 4;
    for (uint32_t k = 0; k < 8; ++k) {
        uint8_t half[2];  // zigzagged signed residuals of pixels 2k and 2k + 1
        for (uint32_t h = 0; h < 2; ++h) {
            half[h] = sy::zigzag(static_cast<int>(idx[2 * k + h]) - (1 << (ib - 1)));
        }
        // Odd bits carry pixel 2k in the first six bytes, even bits in the last two.
        const uint8_t even = k < 6 ? half[1] : half[0];
        const uint8_t odd = k < 6 ? half[0] : half[1];
        uint8_t byte = 0;
        for (uint32_t b = 0; b < 4; ++b) {
            byte = static_cast<uint8_t>(byte | (((even >> b) & 1u) << (2 * b)) | (((odd >> b) & 1u) << (2 * b + 1)));
        }
        r[22 + k] = byte;
    }
    r[30] = static_cast<uint8_t>(mode - 1);
    r[31] = static_cast<uint8_t>(partition);
    return r;
}

void test_bc6h() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Mode 11 (one subset, 10-bit endpoints, 4-bit indices): mode bits 00011, then rw gw bw rx gx
    // bx at 10 bits each from bit 5, then the indices from bit 65 (the first one in 3 bits).
    {
        const uint32_t f[4][3] = {{0x155, 0x2AA, 0x0F0}, {0x3FF, 0, 0x201}, {}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = p == 0 ? 5 : (p * 5 + 1) % 16;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(11, f, ix, 0)});
        CHECK(block_bits(dst.data(), 0, 5) == 3);
        for (uint32_t k = 0; k < 6; ++k) {
            CHECK(block_bits(dst.data(), 5 + 10 * k, 10) == f[k / 3][k % 3]);
        }
        uint32_t at = 65;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = p == 0 ? 3 : 4;
            CHECK(block_bits(dst.data(), at, n) == (ix[p] & ((1u << n) - 1)));
            at += n;
        }
        CHECK(at == 128);
    }
    // Mode 1 (two subsets, 10/5/5/5 bits), partition 17 (second anchor at pixel 2): the scattered
    // header fields at their public positions, then 3-bit indices (2 bits at the anchors).
    {
        const uint32_t f[4][3] = {{0x155, 0x2AA, 0x0F0}, {17, 3, 30}, {1, 2, 3}, {31, 0, 16}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 2) ? p + 1 : (p * 3 + 2) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(1, f, ix, 17)});
        const uint8_t* b = dst.data();
        auto bit = [&](uint32_t at) { return block_bits(b, at, 1); };
        CHECK(block_bits(b, 0, 2) == 0);
        CHECK(bit(2) == (f[2][1] >> 4) && bit(3) == (f[2][2] >> 4) && bit(4) == (f[3][2] >> 4));
        CHECK(block_bits(b, 5, 10) == f[0][0] && block_bits(b, 15, 10) == f[0][1] && block_bits(b, 25, 10) == f[0][2]);
        CHECK(block_bits(b, 35, 5) == f[1][0] && bit(40) == (f[3][1] >> 4) && block_bits(b, 41, 4) == (f[2][1] & 15));
        CHECK(block_bits(b, 45, 5) == f[1][1] && bit(50) == (f[3][2] & 1) && block_bits(b, 51, 4) == (f[3][1] & 15));
        CHECK(block_bits(b, 55, 5) == f[1][2] && bit(60) == ((f[3][2] >> 1) & 1) &&
              block_bits(b, 61, 4) == (f[2][2] & 15));
        CHECK(block_bits(b, 65, 5) == f[2][0] && bit(70) == ((f[3][2] >> 2) & 1) && block_bits(b, 71, 5) == f[3][0]);
        CHECK(bit(76) == ((f[3][2] >> 3) & 1) && block_bits(b, 77, 5) == 17);
        uint32_t at = 82;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = (p == 0 || p == 2) ? 2 : 3;
            CHECK(block_bits(b, at, n) == ix[p]);
            at += n;
        }
        CHECK(at == 128);
    }
    // Mode 14 (one subset, 16-bit base, 4-bit deltas): the base's high bits follow the delta.
    {
        const uint32_t f[4][3] = {{0xABCD, 0x1234, 0xFFFF}, {9, 15, 0}, {}, {}};
        uint32_t ix[16] = {};
        for (uint32_t p = 1; p < 16; ++p) {
            ix[p] = p;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(14, f, ix, 0)});
        const uint8_t* b = dst.data();
        CHECK(block_bits(b, 0, 5) == 15);
        CHECK(block_bits(b, 5, 10) == (f[0][0] & 0x3FF) && block_bits(b, 15, 10) == (f[0][1] & 0x3FF) &&
              block_bits(b, 25, 10) == (f[0][2] & 0x3FF));
        CHECK(block_bits(b, 35, 4) == f[1][0] && block_bits(b, 39, 6) == (f[0][0] >> 10));
        CHECK(block_bits(b, 45, 4) == f[1][1] && block_bits(b, 49, 6) == (f[0][1] >> 10));
        CHECK(block_bits(b, 55, 4) == f[1][2] && block_bits(b, 59, 6) == (f[0][2] >> 10));
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(block_bits(b, 65 + 4 * p - (p > 0 ? 1 : 0), p == 0 ? 3 : 4) == ix[p]);
        }
    }
    // References. Block 0 (mode 11) takes both references from dictionary entry 0 (reference 1);
    // block 1 (mode 1) predicts its endpoints from block 0 and adds +1 to the R of endpoint 1.
    // Dictionary entry 0: endpoint j channel c = (3 j + c) << 10, indices p % 8 - 4 at 4 bits.
    {
        const uint32_t zero_f[4][3] = {};
        const uint32_t zero_ix[16] = {};
        Record b0 = bc6_record(11, zero_f, zero_ix, 0);
        std::fill(b0.begin() + 4, b0.begin() + 30, uint8_t{0});  // no residuals
        b0[2] = 1;  // index reference 1 (merged nibbles 1, 0); endpoint reference = 1 + delta 0
        Record b1(32, 0);
        b1[1] = 1;   // endpoint reference: delta symbol 2 = +1 on index reference 0 -> block 0
        b1[11] = 1;  // R of endpoint 1: residual symbol 2 = +1 (odd byte's bit 0)
        b1[30] = 0;  // mode 1
        b1[31] = 17;
        const std::vector<uint8_t> dst = decode_records(*t, 6, {b0, b1});
        const uint8_t* b = dst.data();
        // Block 0: mode 11, 10-bit fields ((raw ^ 0x8000) >> 6) of the entry's endpoints 0 and 1,
        // where endpoint j channel c holds (3 j + c) << 10, i.e. k << 10 for field k.
        CHECK(block_bits(b, 0, 5) == 3);
        for (uint32_t k = 0; k < 6; ++k) {
            CHECK(block_bits(b, 5 + 10 * k, 10) == (((k << 10) ^ 0x8000u) >> 6));
        }
        // Dictionary index p % 8 - 4 at 4 bits, stored as p % 8 + 4 (3 bits at the anchor).
        uint32_t at = 65;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = p == 0 ? 3 : 4;
            CHECK(block_bits(b, at, n) == (((p % 8) + 4) & ((1u << n) - 1)));
            at += n;
        }
        // Block 1: mode 1. Endpoint 0 as block 0's; the others at 5 bits from the previous
        // block's endpoints (cut to the 5-bit field) plus the residual.
        const uint8_t* c = dst.data() + 16;
        CHECK(block_bits(c, 0, 2) == 0 && block_bits(c, 77, 5) == 17);
        CHECK(block_bits(c, 5, 10) == ((0u ^ 0x8000u) >> 6) && block_bits(c, 15, 10) == (((1u << 10) ^ 0x8000u) >> 6));
        auto five = [](uint32_t raw, uint32_t plus) {
            return ((((raw & 0xF800u) + (plus << 11)) & 0xFFFFu) ^ 0x8000u) >> 11;
        };
        CHECK(block_bits(c, 35, 5) == five(3u << 10, 1));  // rx
        CHECK(block_bits(c, 45, 5) == five(4u << 10, 0));  // gx
        CHECK(block_bits(c, 55, 5) == five(5u << 10, 0));  // bx
        // Block 0 (one subset) left ep2 = ep0 (re-expressed) and ep3 = ep1.
        CHECK(block_bits(c, 65, 5) == five(0u << 10, 0));  // ry
        CHECK(block_bits(c, 71, 5) == five(3u << 10, 0));  // rz
    }
    // Modes outside 1-14 are corrupt.
    for (uint32_t s30 : {14u, 15u, 255u}) {
        Record r(32, 0);
        r[30] = static_cast<uint8_t>(s30);
        bp::DecodeStatus st = bp::DecodeStatus::Ok;
        decode_records(*t, 6, {r}, &st);
        CHECK(st == bp::DecodeStatus::Corrupt);
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

void test_errors() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    std::vector<Record> records;
    for (uint32_t i = 0; i < 64; ++i) {
        Record r(12, 0);
        r[2] = static_cast<uint8_t>(i);
        r[3] = static_cast<uint8_t>(i * 3);
        for (uint32_t k = 4; k < 12; ++k) {
            r[k] = static_cast<uint8_t>(k + i);
        }
        records.push_back(r);
    }
    const std::vector<uint8_t> src = sy::stream(*t, spec_for(4), records);
    std::vector<uint8_t> dst(64 * 8);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Ok);
    CHECK(bp::decode(*t, 4, src.data(), src.size() / 2, dst.data(), dst.size()) == bp::DecodeStatus::Truncated);
    CHECK(bp::decode(*t, 4, nullptr, 0, dst.data(), dst.size()) == bp::DecodeStatus::Truncated);
    std::vector<uint8_t> more(65 * 8);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), more.data(), more.size()) == bp::DecodeStatus::Truncated);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), 12) == bp::DecodeStatus::BadSize);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), 0) == bp::DecodeStatus::BadSize);
    CHECK(bp::decode(*t, 2, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Unsupported);
    CHECK(bp::decode(*t, 2, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Unsupported);
    // Fragment headers promising more payload than there is.
    const std::vector<uint8_t> headers(16, 0x30);
    CHECK(bp::decode(*t, 4, headers.data(), headers.size(), dst.data(), 8) == bp::DecodeStatus::Truncated);
    // BC7 control codes the table leaves unused (0) or makes inconsistent.
    for (uint32_t code : {6u, 5u, 511u}) {
        Record r(32, 0);
        r[30] = static_cast<uint8_t>(code & 31);
        r[31] = static_cast<uint8_t>(code >> 5);
        bp::DecodeStatus st = bp::DecodeStatus::Ok;
        decode_records(*t, 7, {r}, &st);
        CHECK(st == bp::DecodeStatus::Corrupt);
    }
    // Noise never crashes and never writes past the destination.
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (uint32_t round = 0; round < 200; ++round) {
        std::vector<uint8_t> noise(round * 7 % 900);
        for (uint8_t& b : noise) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            b = static_cast<uint8_t>(seed);
        }
        for (uint32_t bc : {1u, 3u, 4u, 5u, 6u, 7u}) {
            std::vector<uint8_t> out(16 * (1 + round % 40) + 16, 0xA5);
            const size_t n = out.size() - 16;
            (void)bp::decode(*t, bc, noise.data(), noise.size(), out.data(), n);
            for (size_t k = n; k < out.size(); ++k) {
                CHECK(out[k] == 0xA5);
            }
        }
    }
}

}  // namespace

int main() {
    test_table_parsing();
    test_tables_file_and_active();
    test_lane_round_trip();
    test_bc1_references();
    test_bc4_bc5_bc3();
    test_bc7_modes();
    test_bc7_references();
    test_bc6h();
    test_errors();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_bcpack: all checks passed\n");
    return 0;
}
