// XMA2 voices (runtime/os/xma_decoder.h) on synthetic streams: packet headers, the frame walk
// across packet headers, padding, skipped packets and buffers, the output ring arithmetic, and
// pump() filling a ring: block order, start skip, input switch, loops and running dry.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "os/xma_decoder.h"

namespace {

int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xma_decoder.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace x = playgta6::os::xma;

constexpr uint32_t kPayload = x::kPacketBits - x::kPacketHeaderBits;

// The test pattern of frame `f`, bit `j` of its body (after the length field).
uint32_t body_bit(uint32_t f, uint32_t j) { return ((f * 7 + j) % 3) == 0 ? 1u : 0u; }

// A synthetic XMA2 stream: frames of the given lengths laid out back to back over the payload of
// the packets listed in `order` (other packets are foreign: a header and junk). A frame never
// starts with 15 bits or fewer left in a packet; those bits stay padding (all ones).
struct Stream {
    std::vector<uint8_t> data;
    std::vector<uint32_t> starts;  // absolute bit positions of the frames
    uint32_t packets = 0;
};

Stream build(const std::vector<uint32_t>& lengths, uint32_t packets, std::vector<uint32_t> order = {}) {
    if (order.empty()) {
        for (uint32_t i = 0; i < packets; ++i) {
            order.push_back(i);
        }
    }
    Stream s;
    s.packets = packets;
    s.data.assign(size_t{packets} * x::kPacketBytes, 0xFF);
    auto abs_of = [&](uint32_t p) { return order[p / kPayload] * x::kPacketBits + 32 + p % kPayload; };
    auto put = [&](uint32_t p, uint32_t b) {
        const uint32_t a = abs_of(p);
        uint8_t& byte = s.data[a / 8];
        const uint8_t m = static_cast<uint8_t>(0x80u >> (a % 8));
        byte = b != 0 ? (byte | m) : (byte & ~m);
    };
    std::vector<uint32_t> logical;
    uint32_t p = 0;
    for (uint32_t f = 0; f < lengths.size(); ++f) {
        if (kPayload - p % kPayload <= x::kFrameLengthBits) {
            p += kPayload - p % kPayload;
        }
        logical.push_back(p);
        s.starts.push_back(abs_of(p));
        for (uint32_t j = 0; j < x::kFrameLengthBits; ++j) {
            put(p + j, (lengths[f] >> (x::kFrameLengthBits - 1 - j)) & 1u);
        }
        for (uint32_t j = x::kFrameLengthBits; j < lengths[f]; ++j) {
            put(p + j, body_bit(f, j - x::kFrameLengthBits));
        }
        p += lengths[f];
    }
    for (uint32_t k = 0; k < packets; ++k) {
        x::PacketHeader h;
        h.first_frame_bit = x::kNoFrame;
        h.skip_packets = 0;
        uint8_t* pkt = s.data.data() + size_t{k} * x::kPacketBytes;
        uint32_t idx = 0;
        while (idx < order.size() && order[idx] != k) {
            ++idx;
        }
        if (idx == order.size()) {  // foreign packet
            h.frame_count = 1;
            h.first_frame_bit = 0;
            std::memset(pkt + 4, 0x5A, x::kPacketBytes - 4);
            x::write_packet_header(pkt, h);
            continue;
        }
        for (uint32_t l : logical) {
            if (l / kPayload == idx) {
                if (h.frame_count == 0) {
                    h.first_frame_bit = l % kPayload;
                }
                ++h.frame_count;
            }
        }
        if (idx + 1 < order.size()) {
            h.skip_packets = order[idx + 1] - order[idx] - 1;
        }
        x::write_packet_header(pkt, h);
    }
    return s;
}

bool frame_bits_match(const x::Frame& f, uint32_t index) {
    if (f.bits_gathered != f.length) {
        return false;
    }
    for (uint32_t j = 0; j < f.length; ++j) {
        const uint32_t b = (f.bits[j / 8] >> (7 - j % 8)) & 1u;
        const uint32_t want = j < x::kFrameLengthBits ? (f.length >> (x::kFrameLengthBits - 1 - j)) & 1u
                                                      : body_bit(index, j - x::kFrameLengthBits);
        if (b != want) {
            return false;
        }
    }
    return true;
}

void test_packet_header() {
    uint8_t p[4] = {};
    x::PacketHeader h{5, 1234, 3, 17};
    x::write_packet_header(p, h);
    // frame count 6 bits, first-frame bit 15, metadata 3, skip 8, big-endian.
    const uint32_t v = (5u << 26) | (1234u << 11) | (3u << 8) | 17u;
    CHECK(p[0] == (v >> 24) && p[1] == ((v >> 16) & 0xFF) && p[2] == ((v >> 8) & 0xFF) && p[3] == (v & 0xFF));
    const x::PacketHeader r = x::parse_packet_header(p);
    CHECK(r.frame_count == 5 && r.first_frame_bit == 1234 && r.metadata == 3 && r.skip_packets == 17);
}

void test_frame_walk() {
    // 9 frames of 4000 bits over 3 packets: frames 4 and 8 cross packet headers.
    const std::vector<uint32_t> lengths(9, 4000);
    Stream s = build(lengths, 3);
    CHECK(x::parse_packet_header(s.data.data()).first_frame_bit == 0);
    CHECK(x::parse_packet_header(s.data.data()).frame_count == 5);
    // Packet 1's first frame starts after the tail of frame 4 (16352 - 4 * 4000 = 352 bits in p0).
    CHECK(x::parse_packet_header(s.data.data() + 2048).first_frame_bit == 4000 - 352);

    auto f = x::read_frame(s.data.data(), s.packets, 0);  // below the payload: the first frame
    uint32_t n = 0;
    while (f) {
        CHECK(f->start == s.starts[n]);
        CHECK(f->length == 4000 && f->complete);
        CHECK(frame_bits_match(*f, n));
        ++n;
        if (!f->has_next) {
            break;
        }
        CHECK(n < s.starts.size() && f->next == s.starts[n]);
        f = x::read_frame(s.data.data(), s.packets, f->next);
    }
    CHECK(n == 9);
}

void test_padding_and_skip() {
    // A frame that would start with <= 15 bits left in packet 0 starts in packet 1 instead.
    Stream s = build({16352 - 10, 300}, 2);
    auto f = x::read_frame(s.data.data(), s.packets, 0);
    CHECK(f && f->has_next && f->next == x::kPacketBits + 32);
    f = x::read_frame(s.data.data(), s.packets, f->next);
    CHECK(f && f->length == 300 && !f->has_next && frame_bits_match(*f, 1));

    // An explicit 0x7FFF after a frame: the rest of the packet is padding.
    Stream q = build({1000}, 1);
    std::vector<uint8_t> two(2 * x::kPacketBytes, 0xFF);
    std::memcpy(two.data(), q.data.data(), x::kPacketBytes);
    Stream r = build({700}, 1);
    std::memcpy(two.data() + x::kPacketBytes, r.data.data(), x::kPacketBytes);
    f = x::read_frame(two.data(), 2, 0);
    CHECK(f && f->length == 1000 && f->has_next && f->next == x::kPacketBits + 32);
    f = x::read_frame(two.data(), 2, f->next);
    CHECK(f && f->length == 700 && !f->has_next);

    // Packet 0 says skip one packet: packet 1 belongs to another stream.
    Stream k = build(std::vector<uint32_t>(5, 4000), 3, {0, 2});
    CHECK(x::parse_packet_header(k.data.data()).skip_packets == 1);
    f = x::read_frame(k.data.data(), k.packets, 0);
    uint32_t n = 0;
    while (f && f->has_next) {
        ++n;
        CHECK(f->next == k.starts[n]);
        f = x::read_frame(k.data.data(), k.packets, f->next);
    }
    CHECK(n == 4 && f && frame_bits_match(*f, 4));
    CHECK(k.starts[4] / x::kPacketBits == 0);  // frame 4 runs on over the foreign packet

    // A length of 15 bits or less is not a frame.
    Stream bad = build({16}, 1);
    bad.data[4] = 0x00;
    bad.data[5] = 0x08;  // length 4
    CHECK(!x::read_frame(bad.data.data(), 1, 0));
}

void test_frame_across_buffers() {
    // Buffer A: frame 0 complete, frame 1 starts 2000 bits before A ends. Buffer B opens with
    // frame 1's remaining bits, then frame 2 at its packet header's first-frame offset.
    Stream whole = build({16352 - 2000, 5000, 3000}, 2);
    std::vector<uint8_t> a(whole.data.begin(), whole.data.begin() + x::kPacketBytes);
    std::vector<uint8_t> b(whole.data.begin() + x::kPacketBytes, whole.data.end());
    auto f0 = x::read_frame(a.data(), 1, 0);
    CHECK(f0 && f0->complete && f0->has_next);
    auto f1 = x::read_frame(a.data(), 1, f0->next);
    CHECK(f1 && !f1->complete && f1->bits_gathered == 2000);
    CHECK(x::continue_frame(b.data(), 1, f1->length - f1->bits_gathered, *f1));
    CHECK(f1->complete && frame_bits_match(*f1, 1));
    const auto first = x::first_frame_from(b.data(), 1, 0);
    CHECK(first && *first == 32 + 3000);
    auto f2 = x::read_frame(b.data(), 1, *first);
    CHECK(f2 && f2->length == 3000 && frame_bits_match(*f2, 2));
}

void test_ring_arithmetic() {
    CHECK(x::ring_step(3, 2, 4) == 1);
    CHECK(x::ring_step(31, 1, 32) == 0);
    CHECK(x::ring_step(7, 70, 5) == (7 % 5 + 70 % 5) % 5);  // indices past the end are reduced
    CHECK(x::ring_room(0, 0, 8, true) == 8);
    CHECK(x::ring_room(0, 0, 8, false) == 0);
    CHECK(x::ring_room(6, 2, 8, true) == 4);
    CHECK(x::ring_room(2, 6, 8, true) == 4);

    uint32_t w[16] = {};
    x::Context c(w);
    CHECK(c.ring_blocks() == 32);  // 0 encodes 32
    w[0] = (4u << 22) | (6u << 27);  // ring of 4, write block left at 6
    CHECK(c.ring_blocks() == 4 && c.write_block() == 2);
    w[9] = 0xABCD0000u | 3u;
    x::advance_read(c, 2);
    CHECK(w[9] == (0xABCD0000u | 1u));
    CHECK(c.output_writable());
    w[1] = 0;
    x::advance_read(c, 0);
    CHECK(!c.output_writable());
}

// Each frame decodes to samples numbered frame * 10000 + sample (per channel, interleaved).
class CountingDecoder final : public x::FrameDecoder {
public:
    bool decode(const x::Frame&, uint32_t channels, uint32_t, int16_t* pcm) override {
        for (uint32_t i = 0; i < x::kFrameSamples; ++i) {
            for (uint32_t ch = 0; ch < channels; ++ch) {
                pcm[i * channels + ch] = static_cast<int16_t>((frame_ % 3) * 10000 + i);
            }
        }
        ++frame_;
        return true;
    }
    uint32_t frame_ = 0;
};

// A fake APU address space: buffers at fixed addresses.
struct Apu {
    std::map<uint32_t, std::vector<uint8_t>*> regions;
    x::ApuMap map() {
        return [this](uint32_t apu, size_t bytes) -> uint8_t* {
            for (auto& [base, v] : regions) {
                if (apu >= base && uint64_t{apu} + bytes <= uint64_t{base} + v->size()) {
                    return v->data() + (apu - base);
                }
            }
            return nullptr;
        };
    }
};

int16_t sample_at(const std::vector<uint8_t>& out, uint32_t block, uint32_t i) {
    const uint32_t o = block * x::kBlockBytes + 2 * i;
    return static_cast<int16_t>(out[o] | (out[o + 1] << 8));
}

void test_pump_mono() {
    Stream s = build(std::vector<uint32_t>(3, 4000), 1);  // 3 frames, 1536 samples, 12 blocks
    std::vector<uint8_t> out(4 * x::kBlockBytes, 0xEE);
    Apu apu;
    apu.regions[0x10000] = &s.data;
    apu.regions[0x20000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (1u << 20) | (4u << 22);  // 1 packet, input 0 valid, ring of 4
    w[1] = (1u << 24) | (3u << 27) | (1u << 31);  // skip 1 subframe at the start, 48 kHz, writable
    w[5] = 0x10000 + 0x10;  // a packet address is aligned down
    w[7] = 0x20000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    CHECK(c.sample_rate() == 48000 && c.channels() == 1);

    CHECK(x::pump(c, v, apu.map()) == 4);
    CHECK(c.write_block() == 0 && !c.output_writable());
    CHECK(sample_at(out, 0, 0) == 128);  // the first subframe was skipped
    CHECK(sample_at(out, 2, 127) == 511 && sample_at(out, 3, 0) == 10000);  // frame 1 in block 3
    CHECK(x::pump(c, v, apu.map()) == 0);  // full until the title takes blocks

    x::advance_read(c, 2);
    CHECK(x::pump(c, v, apu.map()) == 2);
    CHECK(sample_at(out, 0, 0) == 10000 + 128);
    uint32_t total = 6;
    for (int i = 0; i < 10; ++i) {
        x::advance_read(c, 4);
        total += x::pump(c, v, apu.map());
    }
    // 1536 samples - 128 skipped = 11 blocks.
    CHECK(total == 11);
    CHECK(v.ended && v.stats.frames == 3 && v.stats.frames_decoded == 3);
    CHECK(!c.input_valid(0));
    CHECK(c.field(2, 26, 5) == x::kErrorInputExhausted && c.field(2, 31, 1) == 1);

    // More input revives the voice; enable clears the error.
    w[0] |= 1u << 20;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    CHECK(c.field(2, 26, 6) == 0);
    x::advance_read(c, 4);
    CHECK(x::pump(c, v, apu.map()) > 0);
}

void test_pump_stereo_switch_and_span() {
    // Buffer 0: frame 0 and the head of frame 1; buffer 1: the tail of frame 1 and frame 2.
    Stream whole = build({16352 - 2000, 5000, 3000}, 2);
    std::vector<uint8_t> a(whole.data.begin(), whole.data.begin() + x::kPacketBytes);
    std::vector<uint8_t> b(whole.data.begin() + x::kPacketBytes, whole.data.end());
    std::vector<uint8_t> out(32 * x::kBlockBytes, 0);
    Apu apu;
    apu.regions[0x10000] = &a;
    apu.regions[0x30000] = &b;
    apu.regions[0x50000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (3u << 20);  // 1 packet, both inputs valid, ring 0 = 32 blocks
    w[1] = 1u | (1u << 29) | (1u << 31);  // input 1: 1 packet, stereo, writable
    w[5] = 0x10000;
    w[6] = 0x30000;
    w[7] = 0x50000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    // 3 frames * 512 * 2 channels = 3072 samples = 24 blocks.
    CHECK(x::pump(c, v, apu.map()) == 24);
    CHECK(v.stats.frames == 3 && v.stats.bad_frames == 0 && v.stats.input_switches == 1);
    CHECK(c.current_input() == 1 && !c.input_valid(0) && !c.input_valid(1));
    CHECK(sample_at(out, 8, 0) == 10000 && sample_at(out, 8, 1) == 10000);  // frame 1, both channels
    CHECK(c.write_block() == 24 && c.output_writable());
}

void test_pump_loop() {
    // 4 frames; loop once from the end of frame 2 (keep 2 subframes) back to frame 1, whose first
    // subframe is skipped after the jump.
    Stream s = build(std::vector<uint32_t>(4, 3000), 1);
    std::vector<uint8_t> out(32 * x::kBlockBytes, 0);
    Apu apu;
    apu.regions[0x10000] = &s.data;
    apu.regions[0x20000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (1u << 12) | (1u << 20);  // 1 packet, loop count 1, input 0 valid, ring of 32
    w[1] = (1u << 12) | (1u << 17) | (1u << 31);  // loop end subframe 1 (keep 2), loop skip 1
    w[3] = s.starts[1];
    w[4] = s.starts[2] + 100;  // inside frame 2
    w[5] = 0x10000;
    w[7] = 0x20000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    const uint32_t blocks = x::pump(c, v, apu.map());
    // frames 0, 1, 2 (256 kept), then 1 (384 after the skip), 2, 3 (full: loop count is 0).
    const uint32_t samples = 512 + 512 + 256 + 384 + 512 + 512;
    CHECK(v.stats.loops == 1 && v.stats.frames == 6);
    CHECK(c.loop_count() == 0);
    CHECK(blocks == (samples + 127) / 128);
    // After frame 2's cut (block 10 begins at sample 1280), frame 1 again from sample 128.
    CHECK(sample_at(out, 10, 0) == 10000 * (3 % 3) + 128);
}

}  // namespace

int main() {
    test_packet_header();
    test_frame_walk();
    test_padding_and_skip();
    test_frame_across_buffers();
    test_ring_arithmetic();
    test_pump_mono();
    test_pump_stereo_switch_and_span();
    test_pump_loop();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xma_decoder: ok\n");
    return 0;
}
