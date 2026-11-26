// XMA2 voices: see xma_decoder.h.

#include "os/xma_decoder.h"

#include <algorithm>
#include <cstring>

namespace playgta6::os::xma {

namespace {

constexpr uint32_t kPayloadBits = kPacketBits - kPacketHeaderBits;
constexpr uint32_t kPacketAlignMask = kPacketBytes - 1;

uint32_t load_be32(const uint8_t* p) {
    return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | uint32_t{p[3]};
}

// Walks the payload bits of a buffer's packets in stream order: past each packet's header, and
// past the packets of other streams a header says to skip.
class PacketWalker {
public:
    PacketWalker(const uint8_t* data, uint32_t packets, uint32_t at) : data_(data), packets_(packets) {
        packet_ = at / kPacketBits;
        offset_ = std::max(at % kPacketBits, kPacketHeaderBits);
    }

    // Moves off the end of a finished packet; false when the buffer has no packet left.
    bool settle() {
        while (packet_ < packets_ && offset_ >= kPacketBits) {
            const uint32_t skip = parse_packet_header(data_ + size_t{packet_} * kPacketBytes).skip_packets;
            packet_ += 1 + skip;
            offset_ = kPacketHeaderBits;
        }
        return packet_ < packets_;
    }
    bool has_bits() { return settle(); }
    uint32_t left_in_packet() const { return kPacketBits - offset_; }
    uint32_t position() const { return packet_ * kPacketBits + offset_; }
    uint32_t packet() const { return packet_; }

    // One bit, MSB first; call only after has_bits().
    uint32_t bit() {
        const uint8_t byte = data_[size_t{packet_} * kPacketBytes + offset_ / 8];
        const uint32_t b = (byte >> (7 - offset_ % 8)) & 1u;
        ++offset_;
        return b;
    }
    // Up to 32 bits within the current packet, without moving.
    uint32_t peek(uint32_t count) const {
        uint32_t v = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t o = offset_ + i;
            const uint8_t byte = data_[size_t{packet_} * kPacketBytes + o / 8];
            v = (v << 1) | ((byte >> (7 - o % 8)) & 1u);
        }
        return v;
    }

private:
    const uint8_t* data_;
    uint32_t packets_;
    uint32_t packet_ = 0;
    uint32_t offset_ = 0;
};

void put_bit(Frame& f, uint32_t b) {
    if (b != 0) {
        f.bits[f.bits_gathered / 8] |= static_cast<uint8_t>(0x80u >> (f.bits_gathered % 8));
    }
    ++f.bits_gathered;
}

class SilenceDecoder final : public FrameDecoder {
public:
    bool decode(const Frame&, uint32_t channels, uint32_t, int16_t* pcm) override {
        std::fill(pcm, pcm + size_t{kFrameSamples} * channels, int16_t{0});
        return false;
    }
};

}  // namespace

PacketHeader parse_packet_header(const uint8_t* packet) {
    const uint32_t v = load_be32(packet);
    PacketHeader h;
    h.frame_count = v >> 26;
    h.first_frame_bit = (v >> 11) & kNoFrame;
    h.metadata = (v >> 8) & 7u;
    h.skip_packets = v & 0xFFu;
    return h;
}

void write_packet_header(uint8_t* packet, const PacketHeader& h) {
    const uint32_t v = ((h.frame_count & 0x3Fu) << 26) | ((h.first_frame_bit & kNoFrame) << 11) |
                       ((h.metadata & 7u) << 8) | (h.skip_packets & 0xFFu);
    packet[0] = static_cast<uint8_t>(v >> 24);
    packet[1] = static_cast<uint8_t>(v >> 16);
    packet[2] = static_cast<uint8_t>(v >> 8);
    packet[3] = static_cast<uint8_t>(v);
}

uint32_t Context::word(unsigned i) const {
    return static_cast<volatile const uint32_t*>(w_)[i];
}

void Context::set_word(unsigned i, uint32_t v) {
    static_cast<volatile uint32_t*>(w_)[i] = v;
}

uint32_t Context::field(unsigned w, unsigned shift, unsigned bits) const {
    const uint32_t mask = bits >= 32 ? ~0u : ((1u << bits) - 1u);
    return (word(w) >> shift) & mask;
}

void Context::set_field(unsigned w, unsigned shift, unsigned bits, uint32_t v) {
    const uint32_t mask = (bits >= 32 ? ~0u : ((1u << bits) - 1u)) << shift;
    set_word(w, (word(w) & ~mask) | ((v << shift) & mask));
}

uint32_t Context::ring_blocks() const {
    const uint32_t n = field(0, 22, 5);
    return n != 0 ? n : 32u;
}

uint32_t ring_step(uint32_t block, uint32_t advance, uint32_t ring) {
    if (ring == 0) {
        return 0;
    }
    return (block % ring + advance % ring) % ring;
}

uint32_t ring_room(uint32_t write, uint32_t read, uint32_t ring, bool writable) {
    if (!writable || ring == 0) {
        return 0;
    }
    const uint32_t gap = (read % ring + ring - write % ring) % ring;
    return gap == 0 ? ring : gap;
}

std::optional<uint32_t> first_frame_from(const uint8_t* data, uint32_t packets, uint32_t packet) {
    while (packet < packets) {
        const PacketHeader h = parse_packet_header(data + size_t{packet} * kPacketBytes);
        if (h.first_frame_bit < kPayloadBits) {
            return packet * kPacketBits + kPacketHeaderBits + h.first_frame_bit;
        }
        packet += 1 + h.skip_packets;
    }
    return std::nullopt;
}

std::optional<Frame> read_frame(const uint8_t* data, uint32_t packets, uint32_t at) {
    if (data == nullptr || packets == 0) {
        return std::nullopt;
    }
    if (at < kPacketHeaderBits) {
        const auto first = first_frame_from(data, packets, 0);
        if (!first) {
            return std::nullopt;
        }
        at = *first;
    }
    // A position on padding moves to the next packet's first frame (at most once per packet).
    for (uint32_t guard = 0; guard <= packets; ++guard) {
        PacketWalker w(data, packets, at);
        if (!w.has_bits()) {
            return std::nullopt;
        }
        if (w.left_in_packet() <= kFrameLengthBits || w.peek(kFrameLengthBits) == kNoFrame) {
            const PacketHeader h = parse_packet_header(data + size_t{w.packet()} * kPacketBytes);
            const auto next = first_frame_from(data, packets, w.packet() + 1 + h.skip_packets);
            if (!next) {
                return std::nullopt;
            }
            at = *next;
            continue;
        }
        Frame f;
        f.start = w.position();
        f.length = w.peek(kFrameLengthBits);
        if (f.length <= kFrameLengthBits) {
            return std::nullopt;
        }
        f.bits.assign((f.length + 7) / 8, 0);
        while (f.bits_gathered < f.length && w.has_bits()) {
            put_bit(f, w.bit());
        }
        if (f.bits_gathered < f.length) {
            f.complete = false;
            return f;
        }
        // The next frame follows in this packet unless the rest of it is padding.
        if (w.left_in_packet() > kFrameLengthBits && w.peek(kFrameLengthBits) != kNoFrame) {
            f.next = w.position();
            f.has_next = true;
            return f;
        }
        uint32_t following = w.packet();
        if (w.left_in_packet() > 0) {
            following += 1 + parse_packet_header(data + size_t{w.packet()} * kPacketBytes).skip_packets;
        } else if (w.settle()) {
            following = w.packet();
        } else {
            following = packets;
        }
        const auto next = first_frame_from(data, packets, following);
        f.has_next = next.has_value();
        f.next = next.value_or(0);
        return f;
    }
    return std::nullopt;
}

bool continue_frame(const uint8_t* data, uint32_t packets, uint32_t count, Frame& frame) {
    if (data == nullptr || packets == 0) {
        return false;
    }
    PacketWalker w(data, packets, kPacketHeaderBits);
    for (uint32_t i = 0; i < count; ++i) {
        if (frame.bits_gathered >= frame.length || !w.has_bits()) {
            return false;
        }
        put_bit(frame, w.bit());
    }
    frame.complete = frame.bits_gathered == frame.length;
    return true;
}

std::unique_ptr<FrameDecoder> make_silence_decoder() {
    return std::make_unique<SilenceDecoder>();
}

namespace {

const uint8_t* map_input(Context ctx, unsigned buffer, const ApuMap& map, uint32_t& packets) {
    packets = ctx.input_packets(buffer);
    if (packets == 0) {
        return nullptr;
    }
    return map(ctx.input_address(buffer) & ~kPacketAlignMask, size_t{packets} * kPacketBytes);
}

// Decodes `frame` into the voice's PCM queue. `cut` (samples per channel, 0 = all) shortens it.
void emit(Context ctx, Voice& v, const Frame& frame, uint32_t cut) {
    const uint32_t channels = ctx.channels();
    std::vector<int16_t> pcm(size_t{kFrameSamples} * channels);
    ++v.stats.frames;
    if (v.decoder != nullptr && v.decoder->decode(frame, channels, ctx.sample_rate(), pcm.data())) {
        ++v.stats.frames_decoded;
    } else {
        std::fill(pcm.begin(), pcm.end(), int16_t{0});
    }
    const uint32_t keep = cut != 0 ? std::min(cut, kFrameSamples) : kFrameSamples;
    const uint32_t drop = std::min(v.skip, keep);
    v.skip -= drop;
    v.pcm.insert(v.pcm.end(), pcm.begin() + size_t{drop} * channels, pcm.begin() + size_t{keep} * channels);
}

// One step towards more PCM. True: a frame was emitted or the input state moved (call again);
// false: no input to decode now.
bool step(Context ctx, Voice& v, const ApuMap& map) {
    if (v.ended) {
        if (!ctx.input_valid(0) && !ctx.input_valid(1)) {
            return false;
        }
        v.ended = false;  // the title queued more input
    }
    uint32_t cur = ctx.current_input();
    if (!ctx.input_valid(cur)) {
        if (!ctx.input_valid(1 - cur)) {
            v.ended = true;
            v.partial.reset();
            ++v.stats.exhausted;
            ctx.set_error(kErrorInputExhausted);
            return false;
        }
        cur = 1 - cur;
        ctx.set_current_input(cur);
        ctx.set_read_bit(0);
        ++v.stats.input_switches;
    }
    uint32_t packets = 0;
    const uint8_t* data = map_input(ctx, cur, map, packets);
    if (data == nullptr) {
        ctx.set_input_valid(cur, false);
        ++v.stats.bad_frames;
        return true;
    }
    if (v.partial && ctx.read_bit() < kPacketHeaderBits) {
        // The tail of a frame that began in the other buffer opens this one.
        Frame f = std::move(*v.partial);
        v.partial.reset();
        if (continue_frame(data, packets, f.length - f.bits_gathered, f) && f.complete) {
            emit(ctx, v, f, 0);
        } else {
            ++v.stats.bad_frames;
        }
        const auto first = first_frame_from(data, packets, 0);
        if (first) {
            ctx.set_read_bit(*first);
        } else {
            ctx.set_input_valid(cur, false);
        }
        return true;
    }
    std::optional<Frame> frame = read_frame(data, packets, ctx.read_bit());
    if (!frame) {
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
        return true;
    }
    if (!frame->complete) {
        v.partial = std::move(frame);
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
        return true;
    }
    // The loop end lies in this frame: cut it after the loop's end subframe and go back.
    const uint32_t loops = ctx.loop_count();
    const uint32_t loop_end = ctx.loop_end_bit();
    const bool at_loop_end = loops > 0 && loop_end != 0 && loop_end >= frame->start &&
                             (!frame->has_next || loop_end < frame->next);
    if (at_loop_end) {
        emit(ctx, v, *frame, (ctx.loop_end_subframe() + 1) * kSubframeSamples);
        if (loops != kLoopForever) {
            ctx.set_field(0, 12, 8, loops - 1);
        }
        v.skip = ctx.loop_skip_subframes() * kSubframeSamples;
        ctx.set_read_bit(ctx.loop_start_bit());
        ++v.stats.loops;
        return true;
    }
    emit(ctx, v, *frame, 0);
    if (frame->has_next) {
        ctx.set_read_bit(frame->next);
    } else {
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
    }
    return true;
}

}  // namespace

void enable(Context ctx, Voice& voice, std::unique_ptr<FrameDecoder> decoder) {
    ctx.set_field(2, 26, 6, 0);
    voice.decoder = decoder != nullptr ? std::move(decoder) : make_silence_decoder();
    voice.pcm.clear();
    voice.partial.reset();
    voice.ended = false;
    voice.skip = ctx.start_skip_subframes() * kSubframeSamples;
}

void advance_read(Context ctx, uint32_t blocks) {
    ctx.set_read_block(ring_step(ctx.read_block(), blocks, ctx.ring_blocks()));
    if (blocks > 0) {
        ctx.set_output_writable(true);
    }
}

uint32_t pump(Context ctx, Voice& v, const ApuMap& map) {
    uint32_t written = 0;
    const uint32_t ring = ctx.ring_blocks();
    // Bounded: a ring of 32 blocks, a few steps per block.
    for (uint32_t guard = 0; guard < 2 * 32 && ctx.output_writable(); ++guard) {
        for (uint32_t steps = 0; v.pcm.size() < kBlockSamples && steps < 64; ++steps) {
            if (!step(ctx, v, map)) {
                break;
            }
        }
        if (v.pcm.empty() || (v.pcm.size() < kBlockSamples && !v.ended)) {
            break;  // wait for more input (a short tail is written once the stream has ended)
        }
        const uint32_t write = ctx.write_block();
        uint8_t* out = map(ctx.output_address() + write * kBlockBytes, kBlockBytes);
        if (out == nullptr) {
            break;
        }
        for (uint32_t i = 0; i < kBlockSamples; ++i) {
            int16_t s = 0;
            if (!v.pcm.empty()) {
                s = v.pcm.front();
                v.pcm.pop_front();
            }
            const int16_t mag = s == INT16_MIN ? INT16_MAX : static_cast<int16_t>(s < 0 ? -s : s);
            v.stats.peak = std::max(v.stats.peak, mag);
            out[2 * i] = static_cast<uint8_t>(static_cast<uint16_t>(s));
            out[2 * i + 1] = static_cast<uint8_t>(static_cast<uint16_t>(s) >> 8);
        }
        const uint32_t next = (write + 1) % ring;
        ctx.set_write_block(next);
        ++written;
        ++v.stats.blocks_written;
        if (next == ctx.read_block()) {
            ctx.set_output_writable(false);  // full until the title takes blocks
        }
    }
    return written;
}

}  // namespace playgta6::os::xma
