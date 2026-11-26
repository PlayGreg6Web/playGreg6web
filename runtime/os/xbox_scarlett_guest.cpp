#include "os/guest_window.h"
#include "os/xbox_scarlett_guest.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "guest/guest_runtime.h"
#include "os/bcpack.h"
#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xbox_scarlett.h"

namespace playgta6::os::scarlett::guest_api {

using com::GuestPtr;
using com::Guid;
using com::OutPtr;

namespace {

constexpr HResult kOk = com::S_OK_HR;

std::string hex64(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%" PRIx64, v);
    return buf;
}

std::string hr_text(HResult hr) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08" PRIX32, static_cast<uint32_t>(hr));
    return buf;
}

void diag(const std::string& message) {
    com::detail::diagnostic(message.c_str());
}

int64_t trace_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool path_is_audio(const std::u16string& path) {
    std::string lower;
    for (char16_t c : path) {
        lower.push_back(static_cast<char>(c < 0x80 ? std::tolower(static_cast<int>(c)) : '?'));
    }
    return lower.find("audio") != std::string::npos;
}

// PG6_AUDIO_TRACE=1: every request on a file whose path names audio (at most 2000 lines).
void trace_audio_request(const std::u16string& path, const DStorageRequest& r, HResult hr, int64_t enqueued_ms) {
    static std::atomic<uint32_t> lines{0};
    if (!path_is_audio(path) || lines.fetch_add(1) >= 2000) {
        return;
    }
    std::string name;
    for (char16_t c : path) {
        name.push_back(c < 0x80 ? static_cast<char>(c) : '?');
    }
    char line[400];
    std::snprintf(line, sizeof(line),
                  "audio-trace: XCRDAPI '%s' +0x%" PRIx64 " %u -> %u bytes, options 0x%" PRIx64 ", %s, %" PRId64
                  " ms after enqueue (t=%" PRId64 ")",
                  name.c_str(), r.source_offset, r.source_size, r.destination_size, r.options, hr_text(hr).c_str(),
                  trace_now_ms() - enqueued_ms, trace_now_ms());
    diag(line);
}

// One diagnostic per distinct problem (requests can fail by the thousand).
struct OnceFlags {
    std::atomic<bool> bad_options{false};
    std::atomic<bool> bcpack{false};
    std::atomic<bool> bad_pages{false};
    std::atomic<bool> bad_request{false};
    std::atomic<bool> read_failed{false};
    std::atomic<bool> inflate_failed{false};
    std::atomic<bool> bcpack_failed{false};
    std::atomic<bool> bad_signal{false};
    std::atomic<bool> closed_queue{false};
};
OnceFlags& once() {
    static OnceFlags f;
    return f;
}
void diag_once(std::atomic<bool>& flag, const std::string& message) {
    if (!flag.exchange(true, std::memory_order_relaxed)) {
        diag(message);
    }
}

// Reads a NUL-terminated guest string of up to `max_chars` characters (empty if unreadable).
template <typename Ch>
bool read_guest_string(uint64_t va, size_t max_chars, std::basic_string<Ch>* out) {
    out->clear();
    if (va == 0) {
        return false;
    }
    constexpr size_t kChunk = 64;
    Ch buf[kChunk];
    for (size_t done = 0; done < max_chars;) {
        const size_t n = std::min(kChunk, max_chars - done);
        if (!com::read_guest(va + done * sizeof(Ch), buf, n * sizeof(Ch))) {
            // The chunk may run past the end of mapped memory: fall back to one character.
            Ch c{};
            if (!com::read_guest(va + done * sizeof(Ch), &c, sizeof(Ch))) {
                return false;
            }
            if (c == Ch{}) {
                return true;
            }
            out->push_back(c);
            ++done;
            continue;
        }
        for (size_t i = 0; i < n; ++i) {
            if (buf[i] == Ch{}) {
                return true;
            }
            out->push_back(buf[i]);
        }
        done += n;
    }
    return false;  // no terminator within max_chars
}

// ---------------------------------------------------------------------------
// Statistics and tuning
// ---------------------------------------------------------------------------

struct Counters {
    std::atomic<uint64_t> requests_completed{0};
    std::atomic<uint64_t> requests_failed{0};
    std::atomic<uint64_t> requests_cancelled{0};
    std::atomic<uint64_t> bytes_delivered{0};
    std::atomic<uint64_t> file_reads{0};
    std::atomic<uint64_t> coalesced_requests{0};
    std::atomic<uint64_t> decompressed_requests{0};
    std::atomic<uint64_t> submits{0};
    std::atomic<uint64_t> live_queues{0};
    std::atomic<uint64_t> physical_page_requests{0};
    std::atomic<uint64_t> bcpack_requests{0};  // all BCPack requests (option bits 1-4)
    std::atomic<uint64_t> bcpack_decoded{0};   // ... of which unpacked into their destination

    std::atomic<uint32_t> worker_threads{1};
    std::atomic<uint64_t> max_gap{64 * 1024};
    std::atomic<uint64_t> max_span{4 * 1024 * 1024};
    std::atomic<uint64_t> staging_size{32 * 1024 * 1024};
};
Counters& counters() {
    static Counters c;
    return c;
}

// ---------------------------------------------------------------------------
// Inflate (RFC 1950 / RFC 1951)
// ---------------------------------------------------------------------------

constexpr uint16_t kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                      2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr uint16_t kDistBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
constexpr uint8_t kCodeLengthOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

constexpr int kFastBits = 9;

struct Huffman {
    uint16_t count[16]{};
    uint16_t symbol[288]{};
    // Fast table indexed by the next kFastBits input bits (LSB first): (symbol << 4) | length,
    // 0 for codes longer than kFastBits.
    uint16_t fast[1u << kFastBits]{};

    // false for an over-subscribed set of lengths (incomplete sets decode until an unused code).
    bool build(const uint8_t* lengths, int n) {
        std::memset(count, 0, sizeof(count));
        std::memset(fast, 0, sizeof(fast));
        for (int i = 0; i < n; ++i) {
            ++count[lengths[i]];
        }
        if (count[0] == n) {
            return true;  // no codes: decoding anything fails
        }
        int left = 1;
        for (int len = 1; len <= 15; ++len) {
            left <<= 1;
            left -= count[len];
            if (left < 0) {
                return false;
            }
        }
        uint16_t offs[16]{};
        for (int len = 1; len < 15; ++len) {
            offs[len + 1] = static_cast<uint16_t>(offs[len] + count[len]);
        }
        for (int i = 0; i < n; ++i) {
            if (lengths[i] != 0) {
                symbol[offs[lengths[i]]++] = static_cast<uint16_t>(i);
            }
        }
        // Canonical codes, then the reversed (LSB-first) fast entries.
        uint32_t code = 0;
        int index = 0;
        for (int len = 1; len <= 15; ++len) {
            for (int k = 0; k < count[len]; ++k, ++code, ++index) {
                if (len > kFastBits) {
                    continue;
                }
                uint32_t rev = 0;
                for (int b = 0; b < len; ++b) {
                    rev |= ((code >> b) & 1u) << (len - 1 - b);
                }
                const uint16_t entry = static_cast<uint16_t>((symbol[index] << 4) | len);
                for (uint32_t fill = rev; fill < (1u << kFastBits); fill += (1u << len)) {
                    fast[fill] = entry;
                }
            }
            code <<= 1;
        }
        return true;
    }
};

class Inflater {
public:
    Inflater(const uint8_t* src, size_t len, uint8_t* dst, size_t cap)
        : src_(src), len_(len), dst_(dst), cap_(cap) {}

    InflateStatus run() {
        for (;;) {
            uint32_t last = 0;
            uint32_t type = 0;
            if (!bits(1, &last) || !bits(2, &type)) {
                return InflateStatus::Truncated;
            }
            InflateStatus st = InflateStatus::BadData;
            if (type == 0) {
                st = stored();
            } else if (type == 1) {
                st = fixed();
            } else if (type == 2) {
                st = dynamic();
            }
            if (st != InflateStatus::Ok) {
                return st;
            }
            if (last != 0) {
                return InflateStatus::Ok;
            }
        }
    }

    size_t produced() const { return out_; }

    // Discards bits up to the next byte boundary and returns the unread input position.
    size_t byte_position() {
        const int drop = bitcnt_ % 8;
        bitbuf_ >>= drop;
        bitcnt_ -= drop;
        return pos_ - static_cast<size_t>(bitcnt_ / 8);
    }

private:
    bool need(int n) {
        while (bitcnt_ < n) {
            if (pos_ >= len_) {
                return false;
            }
            bitbuf_ |= static_cast<uint64_t>(src_[pos_++]) << bitcnt_;
            bitcnt_ += 8;
        }
        return true;
    }

    bool bits(int n, uint32_t* out) {
        if (n == 0) {
            *out = 0;
            return true;
        }
        if (!need(n)) {
            return false;
        }
        *out = static_cast<uint32_t>(bitbuf_ & ((1ull << n) - 1));
        bitbuf_ >>= n;
        bitcnt_ -= n;
        return true;
    }

    // Symbol or -1 (truncated) / -2 (invalid code).
    int decode(const Huffman& h) {
        need(kFastBits);  // may come up short at the end of the input
        if (bitcnt_ >= kFastBits) {
            const uint16_t e = h.fast[bitbuf_ & ((1u << kFastBits) - 1)];
            if (e != 0) {
                const int len = e & 15;
                bitbuf_ >>= len;
                bitcnt_ -= len;
                return e >> 4;
            }
        }
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= 15; ++len) {
            uint32_t bit = 0;
            if (!bits(1, &bit)) {
                return -1;
            }
            code |= static_cast<int>(bit);
            const int count = h.count[len];
            if (code - count < first) {
                return h.symbol[index + (code - first)];
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        return -2;
    }

    InflateStatus stored() {
        size_t p = byte_position();
        bitbuf_ = 0;
        bitcnt_ = 0;
        if (len_ - p < 4) {
            return InflateStatus::Truncated;
        }
        const uint32_t n = src_[p] | (src_[p + 1] << 8);
        const uint32_t nn = src_[p + 2] | (src_[p + 3] << 8);
        p += 4;
        if (n != (~nn & 0xFFFFu)) {
            return InflateStatus::BadData;
        }
        if (len_ - p < n) {
            return InflateStatus::Truncated;
        }
        if (cap_ - out_ < n) {
            return InflateStatus::OutputTooSmall;
        }
        std::memcpy(dst_ + out_, src_ + p, n);
        out_ += n;
        pos_ = p + n;
        return InflateStatus::Ok;
    }

    InflateStatus codes(const Huffman& lit, const Huffman& dist) {
        for (;;) {
            int sym = decode(lit);
            if (sym < 0) {
                return sym == -1 ? InflateStatus::Truncated : InflateStatus::BadData;
            }
            if (sym < 256) {
                if (out_ == cap_) {
                    return InflateStatus::OutputTooSmall;
                }
                dst_[out_++] = static_cast<uint8_t>(sym);
                continue;
            }
            if (sym == 256) {
                return InflateStatus::Ok;
            }
            sym -= 257;
            if (sym >= 29) {
                return InflateStatus::BadData;
            }
            uint32_t extra = 0;
            if (!bits(kLengthExtra[sym], &extra)) {
                return InflateStatus::Truncated;
            }
            const size_t length = kLengthBase[sym] + extra;
            const int dsym = decode(dist);
            if (dsym < 0) {
                return dsym == -1 ? InflateStatus::Truncated : InflateStatus::BadData;
            }
            if (dsym >= 30) {
                return InflateStatus::BadData;
            }
            if (!bits(kDistExtra[dsym], &extra)) {
                return InflateStatus::Truncated;
            }
            const size_t distance = kDistBase[dsym] + extra;
            if (distance > out_) {
                return InflateStatus::BadData;
            }
            if (cap_ - out_ < length) {
                return InflateStatus::OutputTooSmall;
            }
            uint8_t* d = dst_ + out_;
            const uint8_t* s = d - distance;
            for (size_t i = 0; i < length; ++i) {
                d[i] = s[i];  // overlapping copies repeat the pattern, byte by byte
            }
            out_ += length;
        }
    }

    InflateStatus fixed() {
        static const auto tables = [] {
            std::pair<Huffman, Huffman> t;
            uint8_t lengths[288];
            int i = 0;
            for (; i < 144; ++i) lengths[i] = 8;
            for (; i < 256; ++i) lengths[i] = 9;
            for (; i < 280; ++i) lengths[i] = 7;
            for (; i < 288; ++i) lengths[i] = 8;
            t.first.build(lengths, 288);
            for (i = 0; i < 30; ++i) lengths[i] = 5;
            t.second.build(lengths, 30);
            return t;
        }();
        return codes(tables.first, tables.second);
    }

    InflateStatus dynamic() {
        uint32_t nlen = 0;
        uint32_t ndist = 0;
        uint32_t ncode = 0;
        if (!bits(5, &nlen) || !bits(5, &ndist) || !bits(4, &ncode)) {
            return InflateStatus::Truncated;
        }
        nlen += 257;
        ndist += 1;
        ncode += 4;
        if (nlen > 286 || ndist > 30) {
            return InflateStatus::BadData;
        }
        uint8_t lengths[320]{};
        for (uint32_t i = 0; i < ncode; ++i) {
            uint32_t v = 0;
            if (!bits(3, &v)) {
                return InflateStatus::Truncated;
            }
            lengths[kCodeLengthOrder[i]] = static_cast<uint8_t>(v);
        }
        Huffman lencode;
        if (!lencode.build(lengths, 19)) {
            return InflateStatus::BadData;
        }
        uint32_t index = 0;
        while (index < nlen + ndist) {
            const int sym = decode(lencode);
            if (sym < 0) {
                return sym == -1 ? InflateStatus::Truncated : InflateStatus::BadData;
            }
            if (sym < 16) {
                lengths[index++] = static_cast<uint8_t>(sym);
                continue;
            }
            uint8_t value = 0;
            uint32_t repeat = 0;
            if (sym == 16) {
                if (index == 0) {
                    return InflateStatus::BadData;
                }
                value = lengths[index - 1];
                if (!bits(2, &repeat)) {
                    return InflateStatus::Truncated;
                }
                repeat += 3;
            } else if (sym == 17) {
                if (!bits(3, &repeat)) {
                    return InflateStatus::Truncated;
                }
                repeat += 3;
            } else {
                if (!bits(7, &repeat)) {
                    return InflateStatus::Truncated;
                }
                repeat += 11;
            }
            if (index + repeat > nlen + ndist) {
                return InflateStatus::BadData;
            }
            while (repeat-- != 0) {
                lengths[index++] = value;
            }
        }
        if (lengths[256] == 0) {
            return InflateStatus::BadData;  // no end-of-block code
        }
        Huffman lit;
        Huffman dist;
        if (!lit.build(lengths, static_cast<int>(nlen)) || !dist.build(lengths + nlen, static_cast<int>(ndist))) {
            return InflateStatus::BadData;
        }
        return codes(lit, dist);
    }

    const uint8_t* src_;
    size_t len_;
    uint8_t* dst_;
    size_t cap_;
    size_t pos_ = 0;
    size_t out_ = 0;
    uint64_t bitbuf_ = 0;
    int bitcnt_ = 0;
};

uint32_t adler32(const uint8_t* data, size_t len) {
    uint32_t a = 1;
    uint32_t b = 0;
    while (len != 0) {
        const size_t n = std::min<size_t>(len, 5552);
        for (size_t i = 0; i < n; ++i) {
            a += data[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
        data += n;
        len -= n;
    }
    return (b << 16) | a;
}

// ---------------------------------------------------------------------------
// Host file access (O12 objects)
// ---------------------------------------------------------------------------

Pg6IDStorageFactory* acquire_host_factory() {
    void* p = nullptr;
    if (gdk::failed(pg6_xbox_DStorageGetFactory(&kIidIDStorageFactory, &p))) {
        return nullptr;
    }
    return static_cast<Pg6IDStorageFactory*>(p);
}

// Synchronous reads through the O12 host objects: one host queue + status array per worker.
class HostReader {
public:
    HostReader() {
        factory_ = acquire_host_factory();
        if (factory_ == nullptr) {
            return;
        }
        Pg6DStorageQueueDesc desc{};
        void* q = nullptr;
        void* s = nullptr;
        if (gdk::succeeded(factory_->CreateQueue(&desc, scarlett::kIidIDStorageQueue, &q))) {
            queue_ = static_cast<Pg6IDStorageQueue*>(q);
        }
        if (gdk::succeeded(factory_->CreateStatusArray(1, "pg6 dstorage worker", scarlett::kIidIDStorageStatusArray, &s))) {
            status_ = static_cast<Pg6IDStorageStatusArray*>(s);
        }
    }
    ~HostReader() {
        if (status_ != nullptr) status_->Release();
        if (queue_ != nullptr) queue_->Release();
        if (factory_ != nullptr) factory_->Release();
    }
    HostReader(const HostReader&) = delete;
    HostReader& operator=(const HostReader&) = delete;

    HResult read(Pg6IDStorageFile* file, uint64_t offset, void* dst, uint32_t size) {
        if (queue_ == nullptr || status_ == nullptr) {
            return com::E_FAIL_HR;
        }
        counters().file_reads.fetch_add(1, std::memory_order_relaxed);
        queue_->EnqueueRead(file, offset, dst, size);
        queue_->EnqueueStatus(status_, 0);
        queue_->Submit();
        return status_->GetHResult(0);
    }

private:
    Pg6IDStorageFactory* factory_ = nullptr;
    Pg6IDStorageQueue* queue_ = nullptr;
    Pg6IDStorageStatusArray* status_ = nullptr;
};


// ---------------------------------------------------------------------------
// Bridge objects
// ---------------------------------------------------------------------------

class DStorageStatusArray final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "XCRDAPI.IDStorageStatusArray";

    explicit DStorageStatusArray(uint32_t capacity) : capacity_(capacity), entries_(new Entry[capacity]) {}

    // slot 3
    bool IsComplete(uint32_t index) {
        if (index >= capacity_) {
            diag("XCRDAPI: IsComplete index " + std::to_string(index) + " out of range (capacity " +
                 std::to_string(capacity_) + ")");
            return false;
        }
        return entries_[index].done.load(std::memory_order_acquire);
    }

    // slot 4
    HResult GetHResult(uint32_t index) {
        if (index >= capacity_) {
            return com::E_INVALIDARG_HR;
        }
        const Entry& e = entries_[index];
        return e.done.load(std::memory_order_acquire) ? e.hr.load(std::memory_order_relaxed) : kHrPending;
    }

    uint32_t capacity() const { return capacity_; }

    void mark_pending(uint32_t index) {
        if (index < capacity_) {
            entries_[index].done.store(false, std::memory_order_relaxed);
            entries_[index].hr.store(kHrPending, std::memory_order_relaxed);
        }
    }

    void complete(uint32_t index, HResult hr) {
        if (index < capacity_) {
            entries_[index].hr.store(hr, std::memory_order_relaxed);
            entries_[index].done.store(true, std::memory_order_release);
        }
    }

private:
    struct Entry {
        std::atomic<HResult> hr{kHrPending};
        std::atomic<bool> done{false};
    };
    uint32_t capacity_;
    std::unique_ptr<Entry[]> entries_;
};

class DStorageFile final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "XCRDAPI.IDStorageFile";

    DStorageFile(Pg6IDStorageFile* host, std::u16string path) : host_(host), path_(std::move(path)) {}
    ~DStorageFile() override {
        if (host_ != nullptr) {
            host_->Release();
        }
    }

    // slot 3
    void Close() {
        closed_.store(true, std::memory_order_relaxed);
        host_->Close();
    }

    // slot 4
    HResult GetFileInformation(GuestPtr<ByHandleFileInformation> out) {
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        Pg6DStorageFileInfo info{};
        const HResult hr = host_->GetFileInformation(&info);
        if (gdk::failed(hr)) {
            return hr;
        }
        ByHandleFileInformation bh{};
        bh.file_attributes = 0x1u | 0x80u;  // FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_NORMAL (game data)
        bh.file_size_high = static_cast<uint32_t>(info.file_size >> 32);
        bh.file_size_low = static_cast<uint32_t>(info.file_size);
        bh.number_of_links = 1;
        uint64_t h = 1469598103934665603ull;  // FNV-1a of the path: stable file index
        for (char16_t c : path_) {
            h = (h ^ static_cast<uint16_t>(c)) * 1099511628211ull;
        }
        bh.file_index_high = static_cast<uint32_t>(h >> 32);
        bh.file_index_low = static_cast<uint32_t>(h);
        return out.write(bh) ? kOk : com::E_POINTER_HR;
    }

    Pg6IDStorageFile* host() const { return host_; }
    const std::u16string& path() const { return path_; }

private:
    Pg6IDStorageFile* host_;
    std::u16string path_;
    std::atomic<bool> closed_{false};
};

class DStorageQueue;


// Worker pool executing submitted queue work (one queue is never run by two workers at once).
class Scheduler {
public:
    void post(com::Ref<DStorageQueue> queue);
    void shutdown();
    ~Scheduler();

private:
    void worker();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<com::Ref<DStorageQueue>> ready_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

Scheduler& scheduler() {
    static Scheduler s;
    return s;
}


class DStorageQueue final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "XCRDAPI.IDStorageQueue";

    DStorageQueue(const DStorageQueueDesc& desc, std::string name) : desc_(desc), name_(std::move(name)) {
        capacity_ = desc.capacity;
        counters().live_queues.fetch_add(1, std::memory_order_relaxed);
    }

    ~DStorageQueue() override {
        // Unsubmitted entries never run: their status entries must not stay pending forever.
        for (Item& item : pending_) {
            abort_item(item);
        }
        if (error_event_ != 0) {
            win32::CloseHandle(error_event_);
        }
        counters().live_queues.fetch_sub(1, std::memory_order_relaxed);
    }

    // slot 3
    void EnqueueRequest(GuestPtr<DStorageRequest> request) {
        Item item;
        item.kind = Item::Request;
        if (!request.read(item.req)) {
            diag("XCRDAPI: EnqueueRequest with unreadable request " + hex64(request.va()));
            item.error = com::E_INVALIDARG_HR;
        } else if (!parse_dstorage_options(item.req.options).source_memory) {
            auto file = com::InterfaceArg<DStorageFile>::resolve(item.req.source);
            if (!file) {
                diag_once(once().bad_request, "XCRDAPI: EnqueueRequest source " + hex64(item.req.source) +
                                                  " is not an IDStorageFile");
                item.error = com::E_INVALIDARG_HR;
            } else {
                item.file = std::move(file);
            }
        }
        enqueue(std::move(item));
    }

    // slot 4
    void EnqueueStatus(com::InterfaceArg<DStorageStatusArray> array, uint32_t index) {
        if (!array) {
            diag("XCRDAPI: EnqueueStatus with invalid status array " + hex64(array.va()));
            return;
        }
        if (index >= array->capacity()) {
            diag("XCRDAPI: EnqueueStatus index " + std::to_string(index) + " out of range");
            return;
        }
        Item item;
        item.kind = Item::Status;
        item.status = array;
        item.index = index;
        array->mark_pending(index);
        enqueue(std::move(item));
    }

    // slot 5 (never called by the title): a Win32 event handle is set when every earlier request
    // has completed. The PC API passes an ID3D12Fence* here; anything that is not an event handle
    // is reported and ignored.
    void EnqueueSignal(uint64_t event_or_fence, uint64_t value) {
        Item item;
        item.kind = Item::Signal;
        item.signal = event_or_fence;
        item.value = value;
        enqueue(std::move(item));
    }

    // slot 6
    void Submit() {
        std::lock_guard<std::mutex> lock(mutex_);
        submit_locked();
    }

    // slot 7
    void CancelRequestsWithTag(uint64_t mask, uint64_t value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto mark = [&](Item& item) {
            if (item.kind == Item::Request && (item.req.cancellation_tag & mask) == value) {
                item.cancelled = true;
            }
        };
        std::for_each(pending_.begin(), pending_.end(), mark);
        std::for_each(submitted_.begin(), submitted_.end(), mark);
    }

    // slot 8
    void Close() {
        std::vector<Item> dropped;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
            dropped.swap(pending_);
            cv_.notify_all();
            cv_.wait(lock, [&] { return submitted_.empty() && in_flight_ == 0; });
        }
        for (Item& item : dropped) {
            abort_item(item);
        }
    }

    // slot 9
    uint64_t GetErrorEvent() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_event_ == 0) {
            error_event_ = win32::CreateEventW(nullptr, true, error_.failure_count != 0, nullptr);
        }
        return error_event_;
    }

    // slot 10
    void RetrieveErrorRecord(GuestPtr<DStorageErrorRecord> out) {
        DStorageErrorRecord record{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            record = error_;
            error_ = DStorageErrorRecord{};
            if (error_event_ != 0) {
                win32::ResetEvent(error_event_);
            }
        }
        if (!out.is_null() && !out.write(record)) {
            diag("XCRDAPI: RetrieveErrorRecord with unwritable record " + hex64(out.va()));
        }
    }

    // slot 11
    void Query(GuestPtr<DStorageQueueInfo> out) {
        DStorageQueueInfo info{};
        info.desc = desc_;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t used = occupancy_locked();
            info.empty_slot_count = static_cast<uint16_t>(used >= capacity_ ? 0 : capacity_ - used);
            // Auto-submit happens only when an enqueue finds the queue full.
            info.request_count_until_auto_submit = info.empty_slot_count;
        }
        if (!out.is_null()) {
            out.write(info);
        }
    }

    // Worker side: runs the submitted entries up to and including the next fence. Returns true
    // when more submitted work remains.
    bool run_batch(HostReader& reader);

private:
    struct Item {
        enum Kind : uint8_t { Request, Status, Signal } kind = Request;
        DStorageRequest req{};
        com::InterfaceArg<DStorageFile> file;  // pins the host object (no COM reference)
        com::InterfaceArg<DStorageStatusArray> status;
        uint32_t index = 0;
        uint64_t signal = 0;
        uint64_t value = 0;
        HResult error = kOk;  // found at enqueue time
        bool cancelled = false;
        int64_t enqueued_ms = 0;  // PG6_AUDIO_TRACE: request latency
    };

    size_t occupancy_locked() const { return pending_.size() + submitted_.size() + in_flight_; }

    void abort_item(Item& item) {
        if (item.kind == Item::Status && item.status) {
            item.status->complete(item.index, kHrAbort);
        } else if (item.kind == Item::Request) {
            counters().requests_cancelled.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Work starts only at Submit: the title fills a request's page list after EnqueueRequest and
    // before its own Submit, so running earlier writes into stale pages. The one exception is an
    // enqueue that finds the queue full, which must submit to make progress.
    void enqueue(Item&& item) {
        item.enqueued_ms = trace_now_ms();
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closed_ && occupancy_locked() >= capacity_) {
            if (!pending_.empty()) {
                submit_locked();  // a full queue must make progress
            }
            cv_.wait(lock);
        }
        if (closed_) {
            lock.unlock();
            diag_once(once().closed_queue, "XCRDAPI: enqueue on a closed queue '" + name_ + "'");
            abort_item(item);
            return;
        }
        pending_.push_back(std::move(item));
    }

    void submit_locked() {
        if (closed_ || pending_.empty()) {
            return;
        }
        counters().submits.fetch_add(1, std::memory_order_relaxed);
        for (Item& item : pending_) {
            submitted_.push_back(std::move(item));
        }
        pending_.clear();
        if (!scheduled_) {
            scheduled_ = true;
            scheduler().post(com::Ref<DStorageQueue>::share(this));
        }
    }

    void record_failure(const Item& item, HResult hr, uint32_t command) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_.failure_count++ == 0) {
            error_.hresult = hr;
            error_.command_type = command;
            if (item.kind == Item::Request) {
                error_.request = item.req;
                if (item.file) {
                    const std::u16string& p = item.file->path();
                    const size_t n = std::min<size_t>(p.size(), 259);
                    std::memcpy(error_.filename, p.data(), n * sizeof(char16_t));
                }
                std::string name;
                read_guest_string<char>(item.req.name, 259, &name);
                std::memcpy(error_.request_name, name.data(), std::min<size_t>(name.size(), 259));
            }
        }
        if (error_event_ != 0) {
            win32::SetEvent(error_event_);
        }
    }

    // Executes the requests of one batch (worker thread); returns per-request results.
    void execute(HostReader& reader, std::vector<Item>& items, std::vector<HResult>& results);
    HResult deliver(const Item& item, const uint8_t* data, size_t len, std::vector<uint8_t>& scratch);
    HResult write_destination(const DStorageRequest& r, const uint8_t* data, size_t len);

    DStorageQueueDesc desc_;
    std::string name_;
    size_t capacity_ = 0;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<Item> pending_;
    std::deque<Item> submitted_;
    size_t in_flight_ = 0;
    bool scheduled_ = false;
    bool closed_ = false;
    DStorageErrorRecord error_{};
    uint64_t error_event_ = 0;

    HResult hr_since_fence_ = kOk;  // worker-only (a queue runs on one worker at a time)
};


Scheduler::~Scheduler() {
    shutdown();
}

void Scheduler::post(com::Ref<DStorageQueue> queue) {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_.push_back(std::move(queue));
    const uint32_t want = std::clamp<uint32_t>(counters().worker_threads.load(std::memory_order_relaxed), 1, 16);
    while (threads_.size() < want) {
        threads_.emplace_back([this] { worker(); });
    }
    cv_.notify_one();
}

void Scheduler::worker() {
    HostReader reader;
    for (;;) {
        com::Ref<DStorageQueue> queue;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || !ready_.empty(); });
            if (ready_.empty()) {
                return;  // stopping, nothing left
            }
            queue = std::move(ready_.front());
            ready_.pop_front();
        }
        if (queue->run_batch(reader)) {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_.push_back(std::move(queue));
            cv_.notify_one();
        }
        // Dropping the reference here may destroy the queue (outside the scheduler lock).
    }
}

void Scheduler::shutdown() {
    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        threads.swap(threads_);
    }
    cv_.notify_all();
    for (std::thread& t : threads) {
        t.join();  // workers drain the ready list first
    }
    std::deque<com::Ref<DStorageQueue>> left;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        left.swap(ready_);
        stop_ = false;
    }
}


bool DStorageQueue::run_batch(HostReader& reader) {
    std::vector<Item> batch;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!submitted_.empty()) {
            Item item = std::move(submitted_.front());
            submitted_.pop_front();
            const bool fence = item.kind != Item::Request;
            batch.push_back(std::move(item));
            if (fence) {
                break;
            }
        }
        in_flight_ += batch.size();
    }

    std::vector<HResult> results(batch.size(), kOk);
    execute(reader, batch, results);
    for (size_t i = 0; i < batch.size(); ++i) {
        Item& item = batch[i];
        if (item.kind == Item::Request) {
            if (item.cancelled) {
                counters().requests_cancelled.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            Counters& c = counters();
            c.requests_completed.fetch_add(1, std::memory_order_relaxed);
            if (gdk::failed(results[i])) {
                c.requests_failed.fetch_add(1, std::memory_order_relaxed);
                record_failure(item, results[i], kDStorageCommandRequest);
                if (gdk::succeeded(hr_since_fence_)) {
                    hr_since_fence_ = results[i];
                }
            }
        } else if (item.kind == Item::Status) {
            item.status->complete(item.index, hr_since_fence_);
            hr_since_fence_ = kOk;
        } else {
            if (item.signal == 0 || !win32::SetEvent(item.signal)) {
                diag_once(once().bad_signal, "XCRDAPI: EnqueueSignal target " + hex64(item.signal) +
                                                 " is not a Win32 event (ID3D12Fence signals are not supported)");
            }
        }
    }
    batch.clear();  // drop file / status references before reporting the room

    std::lock_guard<std::mutex> lock(mutex_);
    in_flight_ -= results.size();
    const bool more = !submitted_.empty();
    if (!more) {
        scheduled_ = false;
    }
    cv_.notify_all();
    return more;
}

// Writes `len` bytes to the request's destination: a guest buffer, or with
// DestinationIsPhysicalPages the physical pages listed by the guest array at `destination`.
HResult DStorageQueue::write_destination(const DStorageRequest& r, const uint8_t* data, size_t len) {
    if (!parse_dstorage_options(r.options).physical_pages) {
        return com::write_guest(r.destination, data, len) ? kOk : com::E_INVALIDARG_HR;
    }
    const uint64_t array = r.destination;
    auto entry_at = [array](size_t i, uint64_t* entry) { return com::read_guest(array + i * 8, entry, 8); };
    // destination_reserved: the start offset into the first page (small textures share a page).
    if (!scarlett::xmem_write_physical_pages(entry_at, data, len, r.destination_reserved)) {
        diag_once(once().bad_pages, "XCRDAPI: DestinationIsPhysicalPages request: page array " + hex64(array) +
                                        " names no live physical pages for " + std::to_string(len) + " bytes");
        return com::E_INVALIDARG_HR;
    }
    return kOk;
}

HResult DStorageQueue::deliver(const Item& item, const uint8_t* data, size_t len, std::vector<uint8_t>& scratch) {
    const DStorageRequest& r = item.req;
    const DStorageOptions opts = parse_dstorage_options(r.options);
    if (!opts.compressed()) {
        const HResult hr = write_destination(r, data, len);
        if (gdk::succeeded(hr)) {
            counters().bytes_delivered.fetch_add(len, std::memory_order_relaxed);
        }
        return hr;
    }
    const bool unpack_bc = opts.bcpack_mode != 0;
    // zlib and BCPack together: the console inflates first (to uncompressed_size bytes, the
    // intermediate size), then unpacks the BC blocks into the destination.
    std::vector<uint8_t> intermediate;
    if (unpack_bc && opts.zlib) {
        size_t produced = 0;
        intermediate.resize(r.uncompressed_size);
        const InflateStatus st = r.uncompressed_size == 0
                                     ? InflateStatus::OutputTooSmall
                                     : inflate_stream(data, len, intermediate.data(), intermediate.size(), &produced);
        if (st != InflateStatus::Ok) {
            diag_once(once().inflate_failed, "XCRDAPI: zlib stage of a zlib+BCPack request failed (status " +
                                                 std::to_string(static_cast<int>(st)) + ", intermediate size " +
                                                 std::to_string(r.uncompressed_size) + ")");
            return kHrInvalidData;
        }
        intermediate.resize(produced);
        data = intermediate.data();
        len = intermediate.size();
    }
    const size_t expected =
        unpack_bc ? r.destination_size : (r.uncompressed_size != 0 ? r.uncompressed_size : r.destination_size);
    uint8_t* out = opts.physical_pages ? nullptr : static_cast<uint8_t*>(com::guest_to_host(r.destination, expected));
    if (out == nullptr) {
        scratch.resize(expected);
        out = scratch.data();
    }
    if (unpack_bc) {
        const std::shared_ptr<const bcpack::Tables> tables = bcpack::active_tables();
        const bcpack::DecodeStatus st = tables ? bcpack::decode(*tables, opts.bcpack_mode, data, len, out, expected)
                                               : bcpack::DecodeStatus::Unsupported;
        if (st != bcpack::DecodeStatus::Ok) {
            diag_once(once().bcpack_failed, "XCRDAPI: BCPack request (mode " + std::to_string(opts.bcpack_mode) +
                                                ", " + std::to_string(len) + " -> " + std::to_string(expected) +
                                                " bytes) failed: " + bcpack::decode_status_name(st));
            return kHrInvalidData;
        }
        counters().bcpack_decoded.fetch_add(1, std::memory_order_relaxed);
    } else {
        size_t produced = 0;
        const InflateStatus st = inflate_stream(data, len, out, expected, &produced);
        if (st != InflateStatus::Ok || produced != expected) {
            diag_once(once().inflate_failed, "XCRDAPI: zlib request failed (status " +
                                                 std::to_string(static_cast<int>(st)) + ", produced " +
                                                 std::to_string(produced) + " of " + std::to_string(expected) + ")");
            return kHrInvalidData;
        }
    }
    if (out == scratch.data()) {
        const HResult hr = write_destination(r, out, expected);
        if (gdk::failed(hr)) {
            return hr;
        }
    } else {
        guest_window::mark_written(r.destination, expected);  // O37: inflated in place
    }
    counters().decompressed_requests.fetch_add(1, std::memory_order_relaxed);
    counters().bytes_delivered.fetch_add(expected, std::memory_order_relaxed);
    return kOk;
}

void DStorageQueue::execute(HostReader& reader, std::vector<Item>& items, std::vector<HResult>& results) {
    std::vector<size_t> file_reads;
    std::vector<uint8_t> staging;
    std::vector<uint8_t> scratch;

    for (size_t i = 0; i < items.size(); ++i) {
        Item& item = items[i];
        if (item.kind != Item::Request || item.cancelled) {
            continue;
        }
        if (gdk::failed(item.error)) {
            results[i] = item.error;
            continue;
        }
        const DStorageRequest& r = item.req;
        const DStorageOptions opts = parse_dstorage_options(r.options);
        if (opts.unknown_bits != 0) {
            diag_once(once().bad_options, "XCRDAPI: request options " + hex64(r.options) + " use unsupported bits " +
                                              hex64(opts.unknown_bits));
            results[i] = com::E_NOTIMPL_HR;
            continue;
        }
        if (opts.bcpack_mode != 0) {
            const uint64_t n = counters().bcpack_requests.fetch_add(1, std::memory_order_relaxed) + 1;
            const std::shared_ptr<const bcpack::Tables> tables = bcpack::active_tables();
            if (!tables || !bcpack::supports(*tables, opts.bcpack_mode)) {
                if ((n & (n - 1)) == 0) {  // the 1st, 2nd, 4th, ... (they fail by the hundred: say which)
                    std::string name;
                    read_guest_string(r.name, 96, &name);
                    const std::string why =
                        !tables ? "no BCPack tables are loaded: pass --bcpack-tables <file> or set "
                                  "PLAYGTA6_BCPACK_TABLES (the repository does not ship them; docs/BCPACK.md)"
                                : "the loaded BCPack tables cannot decode mode " + std::to_string(opts.bcpack_mode) +
                                      " (they support: " + bcpack::supported_formats(*tables) +
                                      "; BC2 has no BCPack decoder)";
                    diag("XCRDAPI: request options " + hex64(r.options) + " ask for BCPack (mode " +
                         std::to_string(opts.bcpack_mode) + ", " + std::to_string(r.source_size) + " -> " +
                         std::to_string(r.destination_size) + " bytes, request '" + name + "'); " + why +
                         ", failing with E_NOTIMPL (BCPack request " + std::to_string(n) + ")");
                }
                results[i] = com::E_NOTIMPL_HR;
                continue;
            }
        }
        if (opts.physical_pages) {
            counters().physical_page_requests.fetch_add(1, std::memory_order_relaxed);
        }
        const size_t out_size = opts.bcpack_mode != 0 ? r.destination_size
                                : !opts.zlib
                                    ? r.source_size
                                    : (r.uncompressed_size != 0 ? r.uncompressed_size : r.destination_size);
        if ((out_size != 0 && r.destination == 0) || out_size > r.destination_size) {
            diag_once(once().bad_request, "XCRDAPI: request destination " + hex64(r.destination) + " size " +
                                              std::to_string(r.destination_size) + " cannot take " +
                                              std::to_string(out_size) + " bytes");
            results[i] = com::E_INVALIDARG_HR;
            continue;
        }
        if (r.source_size == 0) {
            continue;  // nothing to transfer
        }
        if (opts.source_memory) {
            staging.resize(r.source_size);
            if (!com::read_guest(r.source, staging.data(), staging.size())) {
                results[i] = com::E_INVALIDARG_HR;
                continue;
            }
            results[i] = deliver(item, staging.data(), staging.size(), scratch);
            continue;
        }
        file_reads.push_back(i);
    }

    // File reads: sorted by (file, offset), nearby ranges of one file read together.
    std::sort(file_reads.begin(), file_reads.end(), [&](size_t a, size_t b) {
        const Item& x = items[a];
        const Item& y = items[b];
        if (x.file.get() != y.file.get()) {
            return x.file.get() < y.file.get();
        }
        return x.req.source_offset < y.req.source_offset;
    });
    const uint64_t max_gap = counters().max_gap.load(std::memory_order_relaxed);
    const uint64_t max_span = std::min(counters().max_span.load(std::memory_order_relaxed),
                                       counters().staging_size.load(std::memory_order_relaxed));

    auto read_single = [&](size_t i) {
        const Item& item = items[i];
        const DStorageRequest& r = item.req;
        const DStorageOptions opts = parse_dstorage_options(r.options);
        const bool plain = !opts.compressed() && !opts.physical_pages;
        void* direct = plain ? com::guest_to_host(r.destination, r.source_size) : nullptr;
        if (direct != nullptr) {
            const HResult hr = reader.read(item.file->host(), r.source_offset, direct, r.source_size);
            guest_window::mark_written(r.destination, static_cast<size_t>(r.source_size));  // O37
            if (gdk::succeeded(hr)) {
                counters().bytes_delivered.fetch_add(r.source_size, std::memory_order_relaxed);
            }
            return hr;
        }
        staging.resize(r.source_size);
        const HResult hr = reader.read(item.file->host(), r.source_offset, staging.data(), r.source_size);
        return gdk::failed(hr) ? hr : deliver(item, staging.data(), r.source_size, scratch);
    };

    size_t g = 0;
    while (g < file_reads.size()) {
        const Item& head = items[file_reads[g]];
        uint64_t start = head.req.source_offset;
        uint64_t end = start + head.req.source_size;
        size_t h = g + 1;
        while (h < file_reads.size() && max_span != 0) {
            const Item& next = items[file_reads[h]];
            if (next.file.get() != head.file.get() || next.req.source_offset > end + max_gap) {
                break;
            }
            const uint64_t new_end = std::max(end, next.req.source_offset + next.req.source_size);
            if (new_end - start > max_span) {
                break;
            }
            end = new_end;
            ++h;
        }
        if (h - g == 1) {
            results[file_reads[g]] = read_single(file_reads[g]);
            g = h;
            continue;
        }
        staging.resize(static_cast<size_t>(end - start));
        const HResult hr = reader.read(head.file->host(), start, staging.data(), static_cast<uint32_t>(end - start));
        if (gdk::failed(hr)) {
            // One of the ranges is bad (e.g. past the end): attribute errors request by request.
            for (size_t k = g; k < h; ++k) {
                results[file_reads[k]] = read_single(file_reads[k]);
            }
        } else {
            counters().coalesced_requests.fetch_add(h - g, std::memory_order_relaxed);
            for (size_t k = g; k < h; ++k) {
                const Item& item = items[file_reads[k]];
                results[file_reads[k]] = deliver(item, staging.data() + (item.req.source_offset - start),
                                                 item.req.source_size, scratch);
            }
        }
        g = h;
    }
    for (size_t i = 0; i < results.size(); ++i) {
        if (gdk::failed(results[i]) && items[i].kind == Item::Request && !items[i].cancelled) {
            diag_once(once().read_failed, "XCRDAPI: request failed with " + hr_text(results[i]));
        }
        if (items[i].kind == Item::Request && items[i].file && audio_trace_enabled()) {
            trace_audio_request(items[i].file->path(), items[i].req, results[i], items[i].enqueued_ms);
        }
    }
}

class DStorageFactory final : public com::ComObject {
public:
    static constexpr const char* kComClassName = "XCRDAPI.IDStorageFactory";

    explicit DStorageFactory(Pg6IDStorageFactory* host) : host_(host) {}
    ~DStorageFactory() override {
        if (host_ != nullptr) {
            host_->Release();
        }
    }

    HResult OpenFile(GuestPtr<char16_t> path, const Guid& riid, OutPtr ppv);
    HResult CreateStatusArray(uint32_t capacity, GuestPtr<char> name, const Guid& riid, OutPtr ppv);
    HResult SetStagingBufferSize(uint32_t size);
    HResult CreateQueue(GuestPtr<DStorageQueueDesc> desc, const Guid& riid, OutPtr ppv);

private:
    Pg6IDStorageFactory* host_;
};

// ---------------------------------------------------------------------------
// Interface declarations (slot tables from the evidence in the header)
// ---------------------------------------------------------------------------


template <size_t N>
struct SlotTable {
    com::MethodDesc slots[N];
};

// Methods for slots 3..3+N-1 from {slot, desc} pairs; gaps are unnamed logging stubs.
template <size_t N>
constexpr SlotTable<N> slot_table(std::initializer_list<std::pair<uint32_t, com::MethodDesc>> entries) {
    SlotTable<N> table{};
    for (const auto& [slot, desc] : entries) {
        table.slots[slot - 3] = desc;
    }
    return table;
}

using FM = com::Methods<DStorageFactory>;
constexpr auto kFactoryTable = slot_table<8>({
    {4, FM::method<&DStorageFactory::OpenFile>("OpenFile")},
    {5, FM::method<&DStorageFactory::CreateStatusArray>("CreateStatusArray")},
    {8, FM::method<&DStorageFactory::SetStagingBufferSize>("SetStagingBufferSize")},
    {10, FM::method<&DStorageFactory::CreateQueue>("CreateQueue")},
});
constexpr auto kIFactory = com::declare_interface<DStorageFactory>("XCRDAPI.IDStorageFactory", kIidDStorageFactory,
                                                                   kFactoryTable.slots, {.slot_count = 24});

using QM = com::Methods<DStorageQueue>;
constexpr com::MethodDesc kQueueMethods[] = {
    QM::method<&DStorageQueue::EnqueueRequest>("EnqueueRequest"),               // 3
    QM::method<&DStorageQueue::EnqueueStatus>("EnqueueStatus"),                 // 4
    QM::method<&DStorageQueue::EnqueueSignal>("EnqueueSignal"),                 // 5
    QM::method<&DStorageQueue::Submit>("Submit"),                               // 6
    QM::method<&DStorageQueue::CancelRequestsWithTag>("CancelRequestsWithTag"), // 7
    QM::method<&DStorageQueue::Close>("Close"),                                 // 8
    QM::method<&DStorageQueue::GetErrorEvent>("GetErrorEvent"),                 // 9
    QM::method<&DStorageQueue::RetrieveErrorRecord>("RetrieveErrorRecord"),     // 10
    QM::method<&DStorageQueue::Query>("Query"),                                 // 11
};
constexpr auto kIQueue = com::declare_interface<DStorageQueue>("XCRDAPI.IDStorageQueue", kIidDStorageQueue,
                                                               kQueueMethods, {.slot_count = 24});

using SM = com::Methods<DStorageStatusArray>;
constexpr com::MethodDesc kStatusMethods[] = {
    SM::method<&DStorageStatusArray::IsComplete>("IsComplete"),  // 3
    SM::method<&DStorageStatusArray::GetHResult>("GetHResult"),  // 4
};
constexpr auto kIStatusArray = com::declare_interface<DStorageStatusArray>(
    "XCRDAPI.IDStorageStatusArray", kIidDStorageStatusArray, kStatusMethods, {.slot_count = 8});

using FiM = com::Methods<DStorageFile>;
constexpr com::MethodDesc kFileMethods[] = {
    FiM::method<&DStorageFile::Close>("Close"),                            // 3
    FiM::method<&DStorageFile::GetFileInformation>("GetFileInformation"),  // 4
};
constexpr auto kIFile =
    com::declare_interface<DStorageFile>("XCRDAPI.IDStorageFile", kIidDStorageFile, kFileMethods, {.slot_count = 8});

bool iid_is(const Guid& riid, const Guid& iid) {
    return riid == iid || riid == com::IID_IUnknown_GUID;
}


HResult DStorageFactory::OpenFile(GuestPtr<char16_t> path, const Guid& riid, OutPtr ppv) {
    if (ppv.is_null() || !ppv.write(0)) {
        return com::E_POINTER_HR;
    }
    if (!iid_is(riid, kIidDStorageFile)) {
        return com::E_NOINTERFACE_HR;
    }
    std::u16string name;
    if (!read_guest_string<char16_t>(path.va(), 32767, &name) || name.empty()) {
        return com::E_INVALIDARG_HR;
    }
    void* raw = nullptr;
    const HResult hr = host_->OpenFile(name.c_str(), scarlett::kIidIDStorageFile, &raw);
    if (audio_trace_enabled()) {
        std::string utf8;
        for (char16_t c : name) {
            utf8.push_back(c < 0x80 ? static_cast<char>(c) : '?');
        }
        diag("audio-trace: XCRDAPI OpenFile '" + utf8 + "' -> " + hr_text(hr));
    }
    if (gdk::failed(hr)) {
        return hr;
    }
    com::Ref<DStorageFile> file =
        com::create_object<DStorageFile>(com::implements(kIFile), static_cast<Pg6IDStorageFile*>(raw), std::move(name));
    if (!file) {
        return com::E_OUTOFMEMORY_HR;
    }
    ppv.write(file.detach(kIFile));
    return kOk;
}

HResult DStorageFactory::CreateStatusArray(uint32_t capacity, GuestPtr<char> name, const Guid& riid, OutPtr ppv) {
    (void)name;
    if (ppv.is_null() || !ppv.write(0)) {
        return com::E_POINTER_HR;
    }
    if (!iid_is(riid, kIidDStorageStatusArray)) {
        return com::E_NOINTERFACE_HR;
    }
    if (capacity == 0 || capacity > (1u << 20)) {
        return com::E_INVALIDARG_HR;
    }
    com::Ref<DStorageStatusArray> array = com::create_object<DStorageStatusArray>(com::implements(kIStatusArray), capacity);
    if (!array) {
        return com::E_OUTOFMEMORY_HR;
    }
    ppv.write(array.detach(kIStatusArray));
    return kOk;
}

HResult DStorageFactory::SetStagingBufferSize(uint32_t size) {
    if (size == 0) {
        return com::E_INVALIDARG_HR;
    }
    counters().staging_size.store(size, std::memory_order_relaxed);
    host_->SetStagingBufferSize(size);  // informational for the host objects; range-checked there
    return kOk;
}

HResult DStorageFactory::CreateQueue(GuestPtr<DStorageQueueDesc> desc_ptr, const Guid& riid, OutPtr ppv) {
    if (ppv.is_null() || !ppv.write(0)) {
        return com::E_POINTER_HR;
    }
    if (!iid_is(riid, kIidDStorageQueue)) {
        return com::E_NOINTERFACE_HR;
    }
    DStorageQueueDesc desc{};
    if (!desc_ptr.read(desc) || desc.capacity == 0 || desc.source_type > 1) {
        return com::E_INVALIDARG_HR;
    }
    std::string name;
    read_guest_string<char>(desc.name, 255, &name);
    com::Ref<DStorageQueue> queue = com::create_object<DStorageQueue>(com::implements(kIQueue), desc, std::move(name));
    if (!queue) {
        return com::E_OUTOFMEMORY_HR;
    }
    ppv.write(queue.detach(kIQueue));
    return kOk;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------


struct ModuleState {
    std::mutex mutex;
    com::Ref<DStorageFactory> factory;
};
ModuleState& module_state() {
    static ModuleState s;
    return s;
}

HResult dstorage_get_factory(const Guid& riid, OutPtr ppv) {
    if (ppv.is_null() || !ppv.write(0)) {
        return com::E_POINTER_HR;
    }
    if (!iid_is(riid, kIidDStorageFactory)) {
        diag("XCRDAPI: DStorageGetFactory for unknown IID " + com::format_guid(riid));
        return com::E_NOINTERFACE_HR;
    }
    ModuleState& st = module_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.factory) {
        Pg6IDStorageFactory* host = acquire_host_factory();
        if (host == nullptr) {
            return com::E_FAIL_HR;
        }
        st.factory = com::create_object<DStorageFactory>(com::implements(kIFactory), host);
        if (!st.factory) {
            return com::E_OUTOFMEMORY_HR;
        }
    }
    com::Ref<DStorageFactory> ref = st.factory;  // the guest's reference
    ppv.write(ref.detach(kIFactory));
    return kOk;
}

struct GuestExport {
    const char* module;
    const char* name;
    uint32_t ordinal;
    playgta6::guest::HostThunkFn thunk;
};

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

InflateStatus inflate_stream(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap, size_t* out_len) {
    *out_len = 0;
    const bool zlib = src_len >= 2 && (src[0] & 0x0F) == 8 && (src[0] >> 4) <= 7 &&
                      ((static_cast<uint32_t>(src[0]) << 8) | src[1]) % 31 == 0;
    if (zlib && (src[1] & 0x20) != 0) {
        return InflateStatus::BadHeader;  // preset dictionary
    }
    const size_t skip = zlib ? 2 : 0;
    Inflater inf(src + skip, src_len - skip, dst, dst_cap);
    const InflateStatus st = inf.run();
    *out_len = inf.produced();
    if (st != InflateStatus::Ok || !zlib) {
        return st;
    }
    const size_t p = skip + inf.byte_position();
    if (src_len - p < 4) {
        return InflateStatus::Truncated;
    }
    const uint32_t expected = (static_cast<uint32_t>(src[p]) << 24) | (static_cast<uint32_t>(src[p + 1]) << 16) |
                              (static_cast<uint32_t>(src[p + 2]) << 8) | src[p + 3];
    return adler32(dst, *out_len) == expected ? InflateStatus::Ok : InflateStatus::ChecksumMismatch;
}

DStorageStats dstorage_stats() {
    const Counters& c = counters();
    DStorageStats s;
    s.requests_completed = c.requests_completed.load();
    s.requests_failed = c.requests_failed.load();
    s.requests_cancelled = c.requests_cancelled.load();
    s.bytes_delivered = c.bytes_delivered.load();
    s.file_reads = c.file_reads.load();
    s.coalesced_requests = c.coalesced_requests.load();
    s.decompressed_requests = c.decompressed_requests.load();
    s.submits = c.submits.load();
    s.live_queues = c.live_queues.load();
    s.physical_page_requests = c.physical_page_requests.load();
    s.bcpack_requests = c.bcpack_requests.load();
    s.bcpack_decoded = c.bcpack_decoded.load();
    return s;
}

void set_dstorage_worker_threads(uint32_t count) {
    counters().worker_threads.store(std::clamp<uint32_t>(count, 1, 16), std::memory_order_relaxed);
}

void set_dstorage_coalescing(uint64_t max_gap, uint64_t max_span) {
    counters().max_gap.store(max_gap, std::memory_order_relaxed);
    counters().max_span.store(max_span, std::memory_order_relaxed);
}

void register_xbox_scarlett_guest_exports() {
    static const GuestExport kExports[] = {
        {"XCRDAPI.dll", "DStorageGetFactory", 0, &pg6_gthunk_dstorage_get_factory},
        {"GameInput.dll", "GameInputCreate", 1, &pg6_gthunk_game_input_create},
        {"xg_xs.dll", "XGCreateTextureComputer", 0, &pg6_gthunk_xg_create_texture_computer},
        {"xg_xs.dll", "XGSuggestSwizzleMode", 0, &pg6_gthunk_xg_suggest_swizzle_mode},
        {"AcpHal.dll", "AcpHalCreate", 0, &pg6_gthunk_acp_hal_create},
        {"chad.dll", "XDspActivate", 0, &pg6_gthunk_xdsp_activate},
        {"chad.dll", "XDspConnectWithMaximumStreamLimit", 0, &pg6_gthunk_xdsp_connect_with_maximum_stream_limit},
        {"chad.dll", "XDspDeactivate", 0, &pg6_gthunk_xdsp_deactivate},
        {"chad.dll", "XDspDisconnect", 0, &pg6_gthunk_xdsp_disconnect},
        {"chad.dll", "XDspSubmitCommand", 0, &pg6_gthunk_xdsp_submit_command},
        {"xapu.dll", "XApuConnect", 0, &pg6_gthunk_xapu_connect},
        {"xapu.dll", "XApuDequeueResult", 0, &pg6_gthunk_xapu_dequeue_result},
        {"xapu.dll", "XApuDisconnect", 0, &pg6_gthunk_xapu_disconnect},
        {"xapu.dll", "XApuEnqueueCommand", 0, &pg6_gthunk_xapu_enqueue_command},
        {"xapu.dll", "XApuGetResultCount", 0, &pg6_gthunk_xapu_get_result_count},
        {"combase.dll", "CoCreateInstance", 0, &pg6_gthunk_co_create_instance},
    };
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    for (const GuestExport& e : kExports) {
        const uint64_t proc = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(e.thunk));
        if (playgta6::guest::guest_callable_for_proc(proc) == 0) {
            playgta6::guest::register_host_callable(e.thunk, e.name, proc);
        }
        win32::register_export(e.module, e.name, e.ordinal, proc);
    }
}

void reset_xbox_scarlett_guest() {
    scheduler().shutdown();
    detail::reset_input_guest();
    reset_audio_endpoint();
    {
        ModuleState& st = module_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.factory.reset();
    }
    Counters& c = counters();
    c.requests_completed = 0;
    c.requests_failed = 0;
    c.requests_cancelled = 0;
    c.bytes_delivered = 0;
    c.file_reads = 0;
    c.coalesced_requests = 0;
    c.decompressed_requests = 0;
    c.submits = 0;
    c.physical_page_requests = 0;
    c.bcpack_requests = 0;
    c.bcpack_decoded = 0;
    c.worker_threads = 1;
    c.max_gap = 64 * 1024;
    c.max_span = 4 * 1024 * 1024;
    c.staging_size = 32 * 1024 * 1024;
    OnceFlags& f = once();
    f.bad_options = false;
    f.bcpack = false;
    f.bad_pages = false;
    f.bad_request = false;
    f.read_failed = false;
    f.inflate_failed = false;
    f.bcpack_failed = false;
    f.bad_signal = false;
    f.closed_queue = false;
}

}  // namespace playgta6::os::scarlett::guest_api

extern "C" {

void pg6_gthunk_dstorage_get_factory(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::dstorage_get_factory>()(state);
}

}  // extern "C"
