#pragma once
// Reusable, safe guest probes (`playgta6_boot --probe SPEC`, `--probe-file FILE`): log each hit of
// a guest function without touching boot.cpp. A spec is
//
//     ADDR[:name][:args=N][:ret][:mem=REG+OFF,LEN][:max=N]
//
//   ADDR       hex guest VA; a main-executable link VA (0x14...) is rebased onto the loaded image,
//              `module.dll+0xOFF` is an offset into a loaded module
//   name       label in the log (letters, digits, `_.-`)
//   args=N     log the first N (0..8) integer arguments: rcx rdx r8 r9, then the stack slots
//   ret        log the return value: the entry line carries "ret":"pending" and a second line
//              ("src":"exit", "entry_seq" = the entry line's seq, "rax", "xmm0") follows when the
//              function returns. Needs the function in the build's `--probe-functions` list (only
//              those get the lifter's exit hook); an unlisted function never gets the exit line.
//   mem=R+O,L  L bytes (1..256) at register R plus signed hex offset O (repeatable, up to 4)
//   max=N      per-probe log line limit (default 200); counters keep counting
//
// Hits come from the dispatch observer (indirect calls) and from the `__playgta6_probe_hit` entry
// counter for functions listed in the build's `--probe-functions` file (direct calls). One JSON
// object per line goes to the probe file. Every guest memory read goes through `safe_read`: a
// range / canonical check first, then the arena's checked copy, page by page on failure, so an
// unmapped, uncommitted or bogus (0x21) address is logged as such and never faults. No game data
// lives here.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace playgta6::guest {

using ProbeReadFn = std::function<bool(uint64_t va, void* dst, size_t len)>;

enum class ReadStatus { Ok, Partial, Unmapped, NullPage, OutOfRange, Overflow };
const char* read_status_name(ReadStatus s);

struct SafeReadResult {
    ReadStatus status = ReadStatus::Unmapped;
    size_t bytes_ok = 0;  // readable prefix of the range (== len for Ok)
};

// Reads [addr, addr + len) into dst without ever faulting: rejects the null page (< 0x10000),
// non-canonical / above-47-bit addresses and wrapping ranges before calling `read`, then tries the
// whole range and, on failure, one 4 KiB page at a time to find the readable prefix.
SafeReadResult safe_read(const ProbeReadFn& read, uint64_t addr, void* dst, size_t len);

struct ProbeMem {
    uint32_t reg = 0;  // GuestReg index
    int64_t off = 0;
    uint32_t len = 0;
};

struct ProbeSpec {
    std::string module;   // empty: `addr` is a VA, else an offset into this module
    uint64_t addr = 0;
    std::string name;
    uint32_t args = 0;
    bool ret = false;
    std::vector<ProbeMem> mem;
    uint64_t max_lines = 200;
};

constexpr uint32_t kProbeMaxArgs = 8;
constexpr uint32_t kProbeMaxMem = 4;
constexpr uint32_t kProbeMaxMemLen = 256;

// False with a message in *error on malformed text.
bool parse_probe_spec(std::string_view text, ProbeSpec* out, std::string* error);
// Register name (rax..r15, rsp, rbp, ...) to its GuestReg index, or -1.
int probe_reg_index(std::string_view name);

// What a hit saw; `regs` is GuestState::gpr.
struct ProbeHit {
    uint64_t seq = 0;
    uint64_t t_ms = 0;
    uint32_t thread = 0;
    const char* source = "dispatch";  // "dispatch" or "entry"
    uint64_t regs[16] = {};
};

// One JSONL line (no trailing newline) for a hit of the probe `spec` at runtime address `addr`.
std::string format_probe_line(const ProbeSpec& spec, uint64_t addr, const ProbeHit& hit, const ProbeReadFn& read);

// Pairs `__playgta6_probe_exit` with its `__playgta6_probe_hit` entry on one thread. A frame is the
// entry's `rsp` (the slot holding the return address); a normal return leaves `rsp` at exactly
// `entry_rsp + 8` (the tail-call-out path too: the callee's `ret` pops our return address). Frames
// of activations that never returned (an unwind or `longjmp` past them) are dropped when a frame
// above them returns or a new activation starts at an address at or above theirs: a live caller's
// frame always has a higher `entry_rsp` than anything it called.
class ProbeFrames {
public:
    struct Frame {
        uint64_t va = 0;
        uint64_t entry_rsp = 0;
        uint64_t entry_seq = 0;  // seq of the logged entry line, 0 when that line was not logged
        long idx = -1;
    };
    static constexpr size_t kMaxDepth = 2048;

    // Records an entry. Stale frames (entry_rsp <= `rsp`) are dropped first; at the depth limit the
    // oldest frame goes.
    void push(long idx, uint64_t va, uint64_t rsp, uint64_t entry_seq) {
        while (!frames_.empty() && frames_.back().entry_rsp <= rsp) {
            frames_.pop_back();
            ++dropped_;
        }
        if (frames_.size() >= kMaxDepth) {
            frames_.erase(frames_.begin());
            ++dropped_;
        }
        frames_.push_back(Frame{va, rsp, entry_seq, idx});
    }
    // Matches an exit of `va` whose `rsp` after the return is `rsp_after`. True with *out set when
    // a frame matches (it and every frame above it are removed); false leaves the stack untouched.
    bool pop(uint64_t va, uint64_t rsp_after, Frame* out) {
        for (size_t i = frames_.size(); i-- > 0;) {
            if (frames_[i].va == va && frames_[i].entry_rsp + 8 == rsp_after) {
                *out = frames_[i];
                dropped_ += frames_.size() - 1 - i;
                frames_.resize(i);
                return true;
            }
            if (frames_[i].entry_rsp > rsp_after) {
                break;  // everything further down is an older, still live caller
            }
        }
        return false;
    }
    size_t depth() const { return frames_.size(); }
    uint64_t dropped() const { return dropped_; }

private:
    std::vector<Frame> frames_;
    uint64_t dropped_ = 0;
};

// One JSONL line (no trailing newline) for the exit of a `ret` probe: `entry_seq` is the entry
// line's seq, or 0 when the exit could not be paired with a logged entry ("paired":false).
std::string format_probe_exit_line(const ProbeSpec& spec, uint64_t addr, uint64_t seq, uint64_t t_ms, uint32_t thread,
                                   uint64_t entry_seq, bool paired, uint64_t rax, const uint64_t xmm0[2]);

class ProbeSet {
public:
    static constexpr size_t kMaxProbes = 64;
    static constexpr uint64_t kDefaultLineBudget = 20000;

    // `specs[i].addr` must already be resolved to a runtime VA. Opens `out_path` (truncating) when
    // non-empty. Duplicate addresses are kept (the first one answers `find`).
    void configure(std::vector<ProbeSpec> specs, const std::string& out_path,
                   uint64_t line_budget = kDefaultLineBudget);
    void clear();
    bool active() const { return count_ != 0; }

    // Probe index for a runtime address, or -1. Cheap range check first (dispatch hot path).
    long find(uint64_t addr) const noexcept;
    size_t size() const { return count_; }
    const ProbeSpec& spec(size_t i) const { return specs_[i]; }
    uint64_t address(size_t i) const { return specs_[i].addr; }

    // Counts a hit. True when it should be logged (under the probe's and the global line limit).
    bool count_hit(size_t i, bool entry) noexcept;
    // Formats and appends the line for a counted hit that count_hit approved. Returns the line's
    // seq (what an exit line names as its entry), 0 when nothing was written.
    uint64_t log_hit(size_t i, ProbeHit hit, const ProbeReadFn& read);
    // Counts an exit of probe `i` and, when it is paired with a logged entry (always) or an
    // unpaired exit within the probe's and the global line limit, writes its exit line.
    void log_exit(size_t i, uint32_t thread, uint64_t t_ms, uint64_t entry_seq, bool paired, uint64_t rax,
                  const uint64_t xmm0[2]) noexcept;
    // An exit whose entry line was rate limited: counted, not logged.
    void count_exit_only(size_t i) noexcept {
        if (i < count_) {
            exit_calls_[i].fetch_add(1, std::memory_order_relaxed);
        }
    }
    uint64_t exit_calls(size_t i) const { return exit_calls_[i].load(std::memory_order_relaxed); }
    uint64_t exit_unpaired(size_t i) const { return exit_unpaired_[i].load(std::memory_order_relaxed); }

    uint64_t calls(size_t i) const { return calls_[i].load(std::memory_order_relaxed); }
    uint64_t logged(size_t i) const { return logged_[i].load(std::memory_order_relaxed); }
    uint64_t entry_calls(size_t i) const { return entry_calls_[i].load(std::memory_order_relaxed); }
    const std::string& out_path() const { return out_path_; }

    // `"guest_probes": {...}` body for the boot report.
    std::string to_json() const;

private:
    std::vector<ProbeSpec> specs_;  // sorted by address
    std::unique_ptr<std::atomic<uint64_t>[]> calls_;
    std::unique_ptr<std::atomic<uint64_t>[]> entry_calls_;
    std::unique_ptr<std::atomic<uint64_t>[]> logged_;
    std::unique_ptr<std::atomic<uint64_t>[]> exit_calls_;
    std::unique_ptr<std::atomic<uint64_t>[]> exit_unpaired_;
    size_t count_ = 0;
    uint64_t min_ = 0;
    uint64_t max_ = 0;
    uint64_t budget_ = kDefaultLineBudget;
    std::atomic<uint64_t> lines_{0};
    std::atomic<uint64_t> seq_{0};
    std::string out_path_;
    std::mutex file_mutex_;
    std::FILE* file_ = nullptr;
    bool file_failed_ = false;

public:
    ~ProbeSet() { clear(); }
    uint64_t next_seq() noexcept { return seq_.fetch_add(1, std::memory_order_relaxed) + 1; }
};

}  // namespace playgta6::guest
