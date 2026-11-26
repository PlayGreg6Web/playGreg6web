// XMA2 voices of the SHAPE XMA block (R40): context fields, the packet / frame walk over the
// input buffers, and the output ring the hardware fills with 16-bit PCM.
//
// This is the host-only half of the XMA path. It knows nothing about guest memory: contexts are
// 16 little-endian words the caller points at, and APU addresses (input / output buffers) go
// through a caller-supplied map. acp_hal_guest.cpp owns the per-context voices, the commands and
// the worker thread that calls pump().
//
// Context words, as the title writes and reads them (the console's XMA context layout):
//   0: input 0 packets (0-11), loop count (12-19, 255 = forever), input 0 / 1 valid (20, 21),
//      output ring size in 256-byte blocks (22-26, 0 = 32), output write block (27-31)
//   1: input 1 packets (0-11), loop end subframe (12-13), loop start subframes to skip (17-19),
//      subframes to skip at the stream start (24-26), sample rate (27-28), stereo (29),
//      output writable (31)
//   2: input read position in bits (0-25), error status (26-30), error set (31)
//   3: loop start in bits (0-25)
//   4: loop end in bits (0-25), current input buffer (31)
//   5, 6: input buffer 0 / 1 APU address; 7: output ring APU address; 8: overlap buffer
//   9: output read block (0-4), advanced by the title (AcpHal command 18)
//
// Input buffers hold 2048-byte XMA2 packets. Each packet starts with a 32-bit big-endian header:
// frame count (6 bits), bit offset of the first frame that starts in this packet, counted from
// the end of the header (15), metadata (3), packets of other streams to skip after this one (8).
// Frames are a 15-bit length in bits (the length field included) and a WMA Pro frame body; they
// run on across packet headers and from one input buffer into the other. A length of 0x7FFF, or
// fewer than 16 bits left in the packet, means the rest of the packet is padding.
//
// Read positions (word 2) and loop points (words 3, 4) are absolute bit offsets in an input
// buffer, packet headers included.
//
// A frame is 512 samples per channel in four 128-sample subframes. FrameDecoder turns its bits
// into PCM; until a WMA Pro decoder exists the default one writes silence, so a voice still runs
// at its real pace: it consumes its input, ends, loops and moves its write block as the hardware
// would.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace playgta6::os::xma {

constexpr uint32_t kPacketBytes = 2048;
constexpr uint32_t kPacketBits = kPacketBytes * 8;
constexpr uint32_t kPacketHeaderBits = 32;
constexpr uint32_t kFrameLengthBits = 15;
constexpr uint32_t kNoFrame = 0x7FFF;  // length / first-frame value meaning "none"
constexpr uint32_t kSubframeSamples = 128;
constexpr uint32_t kFrameSamples = 4 * kSubframeSamples;  // per channel
constexpr uint32_t kBlockBytes = 256;
constexpr uint32_t kBlockSamples = kBlockBytes / 2;  // 16-bit samples, all channels
constexpr uint32_t kContextWords = 16;
constexpr uint32_t kContextBytes = kContextWords * 4;
constexpr uint32_t kLoopForever = 255;
constexpr uint32_t kErrorInputExhausted = 4;  // word 2 error status: both inputs ran out
constexpr uint32_t kSampleRates[4] = {24000, 32000, 44100, 48000};

struct PacketHeader {
    uint32_t frame_count = 0;
    uint32_t first_frame_bit = 0;  // from the end of the header; >= payload bits: none starts here
    uint32_t metadata = 0;
    uint32_t skip_packets = 0;
};
PacketHeader parse_packet_header(const uint8_t* packet);
void write_packet_header(uint8_t* packet, const PacketHeader& h);  // tests and tools

// A view of one context's 16 words (host memory the title shares; every access is volatile).
class Context {
public:
    explicit Context(uint32_t* words) : w_(words) {}
    uint32_t word(unsigned i) const;
    void set_word(unsigned i, uint32_t v);
    uint32_t field(unsigned word, unsigned shift, unsigned bits) const;
    void set_field(unsigned word, unsigned shift, unsigned bits, uint32_t v);

    uint32_t input_packets(unsigned buffer) const { return field(buffer, 0, 12); }
    uint32_t input_address(unsigned buffer) const { return word(5 + buffer); }
    bool input_valid(unsigned buffer) const { return field(0, 20 + buffer, 1) != 0; }
    void set_input_valid(unsigned buffer, bool v) { set_field(0, 20 + buffer, 1, v ? 1 : 0); }
    uint32_t current_input() const { return field(4, 31, 1); }
    void set_current_input(unsigned buffer) { set_field(4, 31, 1, buffer); }
    uint32_t read_bit() const { return field(2, 0, 26); }
    void set_read_bit(uint32_t bit) { set_field(2, 0, 26, bit); }
    uint32_t loop_count() const { return field(0, 12, 8); }
    uint32_t loop_start_bit() const { return field(3, 0, 26); }
    uint32_t loop_end_bit() const { return field(4, 0, 26); }
    uint32_t loop_end_subframe() const { return field(1, 12, 2); }
    uint32_t loop_skip_subframes() const { return field(1, 17, 3); }
    uint32_t start_skip_subframes() const { return field(1, 24, 3); }
    uint32_t sample_rate() const { return kSampleRates[field(1, 27, 2)]; }
    uint32_t channels() const { return 1 + field(1, 29, 1); }
    bool output_writable() const { return field(1, 31, 1) != 0; }
    void set_output_writable(bool v) { set_field(1, 31, 1, v ? 1 : 0); }
    uint32_t output_address() const { return word(7); }
    // Ring size in blocks (0 encodes 32); write / read blocks are reduced modulo it, because the
    // title may leave them at or past the end.
    uint32_t ring_blocks() const;
    uint32_t write_block() const { return field(0, 27, 5) % ring_blocks(); }
    uint32_t read_block() const { return field(9, 0, 5) % ring_blocks(); }
    void set_write_block(uint32_t b) { set_field(0, 27, 5, b % ring_blocks()); }
    void set_read_block(uint32_t b) { set_field(9, 0, 5, b % ring_blocks()); }
    void set_error(uint32_t status) {
        set_field(2, 26, 5, status);
        set_field(2, 31, 1, status != 0 ? 1 : 0);
    }

private:
    uint32_t* w_;
};

// Output ring arithmetic. `advance` is the title's block count, reduced before the sum.
uint32_t ring_step(uint32_t block, uint32_t advance, uint32_t ring);
// Blocks the decoder may still write before its write block meets the read block. A full
// writable ring (write == read) has `ring` blocks of room.
uint32_t ring_room(uint32_t write, uint32_t read, uint32_t ring, bool writable);

// One frame found in an input buffer.
struct Frame {
    uint32_t start = 0;      // bit position of its length field
    uint32_t length = 0;     // bits, the length field included
    uint32_t next = 0;       // bit position of the next frame in this buffer (valid if has_next)
    bool has_next = false;
    bool complete = true;    // false: it runs past the buffer's end (continues in the other one)
    std::vector<uint8_t> bits;  // the frame's bits gathered without packet headers, MSB first
    uint32_t bits_gathered = 0;
};

// The frame at `at` in a buffer of `packets` 2048-byte packets. `at` below the first payload bit
// means "the buffer's first frame". nullopt: no frame starts at or after `at` in this buffer, or
// the length is invalid.
std::optional<Frame> read_frame(const uint8_t* data, uint32_t packets, uint32_t at);
// The bit position of the first frame starting in packet `packet` or a later one; nullopt if none.
std::optional<uint32_t> first_frame_from(const uint8_t* data, uint32_t packets, uint32_t packet);
// Gathers `count` more bits from the start of the buffer's payload into `frame` (the head of a
// frame that started in the other buffer); false if the buffer is too short.
bool continue_frame(const uint8_t* data, uint32_t packets, uint32_t count, Frame& frame);

// Turns one frame's bits into 512 * channels interleaved samples. False: the frame did not
// decode (its samples are silence).
class FrameDecoder {
public:
    virtual ~FrameDecoder() = default;
    virtual bool decode(const Frame& frame, uint32_t channels, uint32_t sample_rate, int16_t* pcm) = 0;
};
// The default: every frame is silence (no WMA Pro decoder yet).
std::unique_ptr<FrameDecoder> make_silence_decoder();

struct VoiceStats {
    uint64_t frames = 0;
    uint64_t frames_decoded = 0;  // FrameDecoder::decode returned true
    uint64_t bad_frames = 0;      // a length or position that walks off the buffer
    uint64_t blocks_written = 0;
    uint64_t loops = 0;
    uint64_t input_switches = 0;
    uint64_t exhausted = 0;       // times both inputs ran dry
    int16_t peak = 0;
};

// Maps an APU address range to host memory (nullptr: unmapped).
using ApuMap = std::function<uint8_t*(uint32_t apu, size_t bytes)>;

// The decoder side of one enabled context.
struct Voice {
    std::unique_ptr<FrameDecoder> decoder;
    std::deque<int16_t> pcm;  // decoded, not yet in the ring
    uint32_t skip = 0;        // samples per channel still to drop
    bool ended = false;       // both inputs ran out
    std::optional<Frame> partial;  // a frame that runs on into the other input buffer
    VoiceStats stats;
};

// Command 5: a fresh voice for the context (clears its error status, takes the start skip).
void enable(Context ctx, Voice& voice, std::unique_ptr<FrameDecoder> decoder);
// Command 18: the title took `blocks` blocks; moves the read block and re-arms "writable".
void advance_read(Context ctx, uint32_t blocks);
// Writes as many blocks as the ring has room and input allows; returns the blocks written.
uint32_t pump(Context ctx, Voice& voice, const ApuMap& map);

}  // namespace playgta6::os::xma
