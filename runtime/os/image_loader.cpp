#include "os/image_loader.h"

#include <algorithm>
#include <cstring>

#include "os/win32_loader.h"

namespace playgta6::os::pgim {

namespace {

constexpr uint32_t kKnownHeaderFlags =
    kFlagHeadersMapped | kFlagExecutableBytesOmitted | kFlagRelocsStripped;
constexpr uint32_t kKnownSectionFlags = kSectionFlagHeaders | kSectionFlagBytesOmitted;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ULL;
constexpr size_t kStreamChunkBytes = 256 * 1024;
constexpr size_t kTableChunkEntries = 64;
constexpr size_t kRelocChunkEntries = 16 * 1024;

ImageError make_error(ImageErrorCode code,
                      uint64_t index = 0,
                      uint64_t offset = 0,
                      uint64_t value = 0,
                      const char* detail = "") {
    ImageError e;
    e.code = code;
    e.index = index;
    e.offset = offset;
    e.value = value;
    e.detail = detail;
    return e;
}

uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
           static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

uint64_t le64(const uint8_t* p) {
    return static_cast<uint64_t>(le32(p)) | static_cast<uint64_t>(le32(p + 4)) << 32;
}

void put_le32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        p[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

void put_le64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

// Strict UTF-8 (rejects overlong forms, surrogates and values above U+10FFFF), like Rust's
// std::str::from_utf8.
bool is_valid_utf8(const uint8_t* s, size_t n) {
    size_t i = 0;
    while (i < n) {
        const uint8_t b = s[i];
        if (b < 0x80) {
            ++i;
            continue;
        }
        size_t len = 0;
        uint32_t cp = 0;
        uint32_t min = 0;
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
            cp = b & 0x1Fu;
            min = 0x80;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            cp = b & 0x0Fu;
            min = 0x800;
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            cp = b & 0x07u;
            min = 0x10000;
        } else {
            return false;
        }
        if (i + len > n) {
            return false;
        }
        for (size_t k = 1; k < len; ++k) {
            if ((s[i + k] & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (s[i + k] & 0x3Fu);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += len;
    }
    return true;
}

bool read_exact(ImageReader& reader, uint64_t offset, void* dst, size_t len) {
    return len == 0 || reader.read(offset, dst, len);
}

// Everything read_image checks once all tables are decoded (GuestImage::validate in Rust).
ImageError validate_layout(const ImageLayout& layout) {
    const ImageHeader& h = layout.header;
    if (h.image_base > UINT64_MAX - h.size_of_image) {
        return make_error(ImageErrorCode::BadHeaderField, 0, 24, h.image_base,
                          "image_base + size_of_image");
    }
    const bool first_is_headers = !layout.sections.empty() && layout.sections[0].is_headers();
    if (h.headers_mapped() != first_is_headers) {
        return make_error(ImageErrorCode::InvalidSection, 0, 0, 0,
                          "headers flag does not match the first section");
    }

    const bool omit = h.executable_bytes_omitted();
    for (size_t index = 0; index < layout.sections.size(); ++index) {
        const SectionEntry& s = layout.sections[index];
        auto invalid = [&](const char* reason) {
            return make_error(ImageErrorCode::InvalidSection, index, s.rva, 0, reason);
        };
        const bool exec = (s.protection & kProtExecute) != 0;
        if (s.virtual_size == 0) {
            return invalid("zero in-memory size");
        }
        if (s.payload_length > s.virtual_size) {
            return invalid("initialised bytes exceed in-memory size");
        }
        if (s.is_headers() && (index != 0 || s.rva != 0)) {
            return invalid("headers pseudo-section must be first and at RVA 0");
        }
        if (s.bytes_omitted() && !(omit && exec && s.payload_length == 0)) {
            return invalid("omitted bytes only allowed for empty executable sections");
        }
        if (omit && exec && !s.is_headers() && !s.bytes_omitted()) {
            return invalid("executable section bytes must be omitted");
        }
        if (s.end() > h.size_of_image) {
            return make_error(ImageErrorCode::SectionOutOfImage, index, s.rva, s.virtual_size);
        }
        if (index > 0 && layout.sections[index - 1].end() > s.rva) {
            return make_error(ImageErrorCode::OverlappingSections, index, s.rva);
        }
    }

    // Relocation sites in ascending RVA order across both lists.
    struct Site {
        uint32_t rva;
        uint32_t width;
    };
    size_t d = 0;
    size_t l = 0;
    bool have_prev = false;
    Site prev{0, 0};
    while (d < layout.dir64.size() || l < layout.highlow.size()) {
        Site cur;
        if (l >= layout.highlow.size() ||
            (d < layout.dir64.size() && layout.dir64[d] <= layout.highlow[l])) {
            cur = {layout.dir64[d++], 8};
        } else {
            cur = {layout.highlow[l++], 4};
        }
        if (have_prev) {
            if (prev.rva >= cur.rva) {
                return make_error(ImageErrorCode::RelocationsUnsorted, 0, cur.rva);
            }
            if (static_cast<uint64_t>(prev.rva) + prev.width > cur.rva) {
                return make_error(ImageErrorCode::OverlappingRelocations, 0, cur.rva, prev.rva);
            }
        }
        have_prev = true;
        prev = cur;

        // The site must lie in mapped sections whose bytes are stored (contiguous if it
        // straddles two).
        const uint64_t site_end = static_cast<uint64_t>(cur.rva) + cur.width;
        if (site_end > h.size_of_image) {
            return make_error(ImageErrorCode::RelocSiteOutOfRange, 0, cur.rva, h.size_of_image);
        }
        auto section_at = [&](uint64_t rva) -> const SectionEntry* {
            auto it = std::upper_bound(
                layout.sections.begin(), layout.sections.end(), rva,
                [](uint64_t v, const SectionEntry& s) { return v < s.rva; });
            if (it == layout.sections.begin()) {
                return nullptr;
            }
            --it;
            return rva < it->end() ? &*it : nullptr;
        };
        const SectionEntry* first = section_at(cur.rva);
        const SectionEntry* last = section_at(site_end - 1);
        const bool ok = first != nullptr && last != nullptr && !first->bytes_omitted() &&
                        !last->bytes_omitted() && (first == last || first->end() == last->rva);
        if (!ok) {
            return make_error(ImageErrorCode::RelocSiteUnmapped, 0, cur.rva);
        }
    }
    return {};
}

ImageError read_reloc_list(ImageReader& reader,
                           uint64_t table_offset,
                           uint32_t count,
                           std::vector<uint32_t>& out) {
    std::vector<uint8_t> buf;
    uint64_t done = 0;
    while (done < count) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(kRelocChunkEntries, count - done));
        buf.resize(n * 4);
        const uint64_t at = table_offset + done * 4;
        if (!read_exact(reader, at, buf.data(), buf.size())) {
            return make_error(ImageErrorCode::ReadFailed, 0, at);
        }
        for (size_t i = 0; i < n; ++i) {
            const uint32_t rva = le32(buf.data() + i * 4);
            if (!out.empty() && out.back() >= rva) {
                return make_error(ImageErrorCode::RelocationsUnsorted, 0, rva);
            }
            out.push_back(rva);
        }
        done += n;
    }
    return {};
}

}  // namespace

const char* error_name(ImageErrorCode code) {
    switch (code) {
        case ImageErrorCode::Ok: return "Ok";
        case ImageErrorCode::Truncated: return "Truncated";
        case ImageErrorCode::BadMagic: return "BadMagic";
        case ImageErrorCode::UnsupportedVersion: return "UnsupportedVersion";
        case ImageErrorCode::BadHeaderField: return "BadHeaderField";
        case ImageErrorCode::SectionOutOfImage: return "SectionOutOfImage";
        case ImageErrorCode::OverlappingSections: return "OverlappingSections";
        case ImageErrorCode::InvalidSection: return "InvalidSection";
        case ImageErrorCode::PayloadOutOfRange: return "PayloadOutOfRange";
        case ImageErrorCode::PayloadMisaligned: return "PayloadMisaligned";
        case ImageErrorCode::PayloadOverlap: return "PayloadOverlap";
        case ImageErrorCode::ChecksumMismatch: return "ChecksumMismatch";
        case ImageErrorCode::RelocationsUnsorted: return "RelocationsUnsorted";
        case ImageErrorCode::OverlappingRelocations: return "OverlappingRelocations";
        case ImageErrorCode::RelocSiteOutOfRange: return "RelocSiteOutOfRange";
        case ImageErrorCode::RelocSiteUnmapped: return "RelocSiteUnmapped";
        case ImageErrorCode::ReadFailed: return "ReadFailed";
        case ImageErrorCode::InvalidArgument: return "InvalidArgument";
        case ImageErrorCode::RelocationsStripped: return "RelocationsStripped";
        case ImageErrorCode::ReserveFailed: return "ReserveFailed";
        case ImageErrorCode::CommitFailed: return "CommitFailed";
        case ImageErrorCode::WriteFailed: return "WriteFailed";
        case ImageErrorCode::ProtectFailed: return "ProtectFailed";
        case ImageErrorCode::RegisterFailed: return "RegisterFailed";
    }
    return "Unknown";
}

bool MemoryImageReader::read(uint64_t offset, void* dst, size_t len) {
    if (offset > size_ || len > size_ - static_cast<size_t>(offset)) {
        return false;
    }
    if (len != 0) {
        std::memcpy(dst, data_ + static_cast<size_t>(offset), len);
    }
    return true;
}

bool FunctionImageReader::read(uint64_t offset, void* dst, size_t len) {
    if (offset > size_ || len > size_ - offset) {
        return false;
    }
    return !fn_ || fn_(offset, dst, len);
}

uint64_t fnv1a_64(const uint8_t* data, size_t len, uint64_t state) {
    for (size_t i = 0; i < len; ++i) {
        state ^= data[i];
        state *= kFnvPrime;
    }
    return state;
}

ImageError parse_layout(ImageReader& reader, ImageLayout& out) {
    out = ImageLayout{};
    const uint64_t file_len = reader.size();
    if (file_len < kHeaderBytes) {
        return make_error(ImageErrorCode::Truncated, 0, kHeaderBytes, file_len, "image header");
    }
    uint8_t hdr[kHeaderBytes];
    if (!reader.read(0, hdr, sizeof(hdr))) {
        return make_error(ImageErrorCode::ReadFailed, 0, 0);
    }
    if (std::memcmp(hdr, "PGIM", 4) != 0) {
        return make_error(ImageErrorCode::BadMagic, 0, 0, le32(hdr));
    }
    const uint32_t version = le32(hdr + 4);
    if (version != kVersion) {
        return make_error(ImageErrorCode::UnsupportedVersion, 0, 4, version);
    }
    struct Expect {
        size_t at;
        const char* field;
        uint32_t expected;
    };
    const Expect expects[] = {
        {8, "header_bytes", static_cast<uint32_t>(kHeaderBytes)},
        {12, "section_entry_bytes", static_cast<uint32_t>(kSectionEntryBytes)},
        {20, "page_size", static_cast<uint32_t>(kPayloadAlignment)},
    };
    for (const Expect& e : expects) {
        if (le32(hdr + e.at) != e.expected) {
            return make_error(ImageErrorCode::BadHeaderField, 0, e.at, le32(hdr + e.at), e.field);
        }
    }
    ImageHeader& h = out.header;
    h.flags = le32(hdr + 16);
    if ((h.flags & ~kKnownHeaderFlags) != 0) {
        return make_error(ImageErrorCode::BadHeaderField, 0, 16, h.flags, "flags");
    }
    h.image_base = le64(hdr + 24);
    h.size_of_image = le32(hdr + 32);
    h.section_alignment = le32(hdr + 36);
    h.entry_point_rva = le32(hdr + 40);
    h.section_count = le32(hdr + 44);
    h.dir64_count = le32(hdr + 48);
    h.highlow_count = le32(hdr + 52);
    h.file_size = le64(hdr + 56);
    if (h.file_size != file_len) {
        return make_error(ImageErrorCode::BadHeaderField, 0, 56, h.file_size, "file_size");
    }

    // All counts are u32, so these products cannot overflow u64.
    const uint64_t table_start = kHeaderBytes + uint64_t{h.section_count} * kSectionEntryBytes;
    const uint64_t tables_end =
        table_start + (uint64_t{h.dir64_count} + h.highlow_count) * 4;
    if (tables_end > file_len) {
        return make_error(ImageErrorCode::Truncated, 0, tables_end, file_len,
                          "section and relocation tables");
    }

    struct Payload {
        uint64_t offset;
        uint64_t end;
        uint64_t index;
    };
    std::vector<Payload> payloads;
    std::vector<uint8_t> chunk;
    for (uint64_t index = 0; index < h.section_count;) {
        const size_t n = static_cast<size_t>(
            std::min<uint64_t>(kTableChunkEntries, h.section_count - index));
        chunk.resize(n * kSectionEntryBytes);
        const uint64_t chunk_at = kHeaderBytes + index * kSectionEntryBytes;
        if (!read_exact(reader, chunk_at, chunk.data(), chunk.size())) {
            return make_error(ImageErrorCode::ReadFailed, index, chunk_at);
        }
        for (size_t k = 0; k < n; ++k, ++index) {
            const uint8_t* entry = chunk.data() + k * kSectionEntryBytes;
            auto invalid = [&](const char* reason) {
                return make_error(ImageErrorCode::InvalidSection, index, 0, 0, reason);
            };
            SectionEntry s;
            size_t name_len = 0;
            while (name_len < kSectionNameBytes && entry[name_len] != 0) {
                ++name_len;
            }
            for (size_t i = name_len; i < kSectionNameBytes; ++i) {
                if (entry[i] != 0) {
                    return invalid("name padding is not zero");
                }
            }
            if (!is_valid_utf8(entry, name_len)) {
                return invalid("name is not UTF-8");
            }
            s.name.assign(reinterpret_cast<const char*>(entry), name_len);
            s.rva = le32(entry + 16);
            s.virtual_size = le32(entry + 20);
            s.protection = le32(entry + 24);
            if ((s.protection & ~7u) != 0) {
                return invalid("unknown protection bits");
            }
            s.flags = le32(entry + 28);
            if ((s.flags & ~kKnownSectionFlags) != 0) {
                return invalid("unknown section flags");
            }
            if (le32(entry + 60) != 0) {
                return invalid("reserved field is not zero");
            }
            s.payload_offset = le64(entry + 32);
            s.payload_length = le64(entry + 40);
            s.checksum = le64(entry + 48);
            s.characteristics = le32(entry + 56);

            if (s.payload_length == 0) {
                if (s.payload_offset != 0) {
                    return invalid("empty payload with non-zero offset");
                }
                if (s.checksum != fnv1a_64(nullptr, 0)) {
                    return make_error(ImageErrorCode::ChecksumMismatch, index, 0,
                                      fnv1a_64(nullptr, 0));
                }
            } else {
                if (s.payload_offset % kPayloadAlignment != 0) {
                    return make_error(ImageErrorCode::PayloadMisaligned, index, s.payload_offset);
                }
                const bool wraps = s.payload_length > UINT64_MAX - s.payload_offset;
                const uint64_t end = wraps ? 0 : s.payload_offset + s.payload_length;
                if (wraps || end > file_len || s.payload_offset < tables_end) {
                    return make_error(ImageErrorCode::PayloadOutOfRange, index, s.payload_offset,
                                      s.payload_length);
                }
                payloads.push_back({s.payload_offset, end, index});
            }
            out.sections.push_back(std::move(s));
        }
    }

    std::sort(payloads.begin(), payloads.end(), [](const Payload& a, const Payload& b) {
        return a.offset != b.offset ? a.offset < b.offset : a.end < b.end;
    });
    for (size_t i = 1; i < payloads.size(); ++i) {
        if (payloads[i - 1].end > payloads[i].offset) {
            return make_error(ImageErrorCode::PayloadOverlap, payloads[i].index,
                              payloads[i].offset);
        }
    }

    ImageError err = read_reloc_list(reader, table_start, h.dir64_count, out.dir64);
    if (!err.ok()) {
        return err;
    }
    err = read_reloc_list(reader, table_start + uint64_t{h.dir64_count} * 4, h.highlow_count,
                          out.highlow);
    if (!err.ok()) {
        return err;
    }
    return validate_layout(out);
}

ImageError parse_image(std::span<const uint8_t> bytes, ParsedImage& out) {
    out = ParsedImage{};
    MemoryImageReader reader(bytes);
    ImageError err = parse_layout(reader, out.layout);
    if (!err.ok()) {
        return err;
    }
    out.payloads.reserve(out.layout.sections.size());
    for (size_t i = 0; i < out.layout.sections.size(); ++i) {
        const SectionEntry& s = out.layout.sections[i];
        if (s.payload_length == 0) {
            out.payloads.emplace_back();
            continue;
        }
        // parse_layout bounded offset + length by the container size.
        const std::span<const uint8_t> view =
            bytes.subspan(static_cast<size_t>(s.payload_offset), static_cast<size_t>(s.payload_length));
        const uint64_t actual = fnv1a_64(view.data(), view.size());
        if (actual != s.checksum) {
            return make_error(ImageErrorCode::ChecksumMismatch, i, s.payload_offset, actual);
        }
        out.payloads.push_back(view);
    }
    return {};
}

PageProtect page_protect_for(uint32_t protection) {
    const bool write = (protection & kProtWrite) != 0;
    const bool exec = (protection & kProtExecute) != 0;
    const bool read = (protection & kProtRead) != 0 || write;
    if (write && exec) {
        return PageProtect::ExecuteReadWrite;
    }
    if (exec) {
        return PageProtect::ExecuteRead;
    }
    if (write) {
        return PageProtect::ReadWrite;
    }
    return read ? PageProtect::ReadOnly : PageProtect::NoAccess;
}

std::vector<PageRun> page_protections(const ImageLayout& layout) {
    struct Span {
        uint64_t start;
        uint64_t end;
        uint32_t protection;
    };
    std::vector<Span> spans;
    spans.reserve(layout.sections.size());
    std::vector<uint64_t> cuts;
    cuts.reserve(layout.sections.size() * 2);
    for (const SectionEntry& s : layout.sections) {
        const uint64_t start = align_down(s.rva, kPageSize);
        uint64_t end = 0;
        align_up_checked(s.end(), kPageSize, end);
        spans.push_back({start, end, s.protection});
        cuts.push_back(start);
        cuts.push_back(end);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    // Sections are sorted and disjoint, so span starts and ends are both non-decreasing and the
    // spans covering a cut form a window that only moves forward.
    std::vector<PageRun> runs;
    size_t lo = 0;
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        const uint64_t start = cuts[c];
        const uint64_t end = cuts[c + 1];
        while (lo < spans.size() && spans[lo].end <= start) {
            ++lo;
        }
        bool covered = false;
        uint32_t protection = 0;
        for (size_t i = lo; i < spans.size() && spans[i].start <= start; ++i) {
            covered = true;
            protection |= spans[i].protection;
        }
        if (!covered) {
            continue;
        }
        if (!runs.empty() && runs.back().rva + runs.back().size == start &&
            runs.back().protection == protection) {
            runs.back().size += end - start;
        } else {
            runs.push_back({start, end - start, protection});
        }
    }
    return runs;
}

ImageError load_image(ImageReader& reader,
                      VirtualMemoryArena& arena,
                      const LoadOptions& options,
                      LoadedImage& out) {
    out = LoadedImage{};
    ImageLayout layout;
    ImageError err = parse_layout(reader, layout);
    if (!err.ok()) {
        return err;
    }
    const ImageHeader& h = layout.header;
    if (h.size_of_image == 0) {
        return make_error(ImageErrorCode::ReserveFailed, 0, 0, 0, "SizeOfImage is zero");
    }

    const bool can_rebase = !h.relocs_stripped();
    const uint64_t requested = options.requested_base != 0 ? options.requested_base : h.image_base;
    if (requested != h.image_base && !can_rebase) {
        return make_error(ImageErrorCode::RelocationsStripped, 0, 0, requested,
                          "relocations stripped; base must equal the preferred base");
    }

    uint64_t base = arena.reserve(requested, h.size_of_image, PageProtect::NoAccess);
    if (base == 0 && options.allow_fallback && can_rebase) {
        base = arena.reserve(0, h.size_of_image, PageProtect::NoAccess);
    }
    if (base == 0) {
        return make_error(ImageErrorCode::ReserveFailed, 0, 0, requested);
    }
    auto fail = [&](ImageError e) {
        arena.release(base);
        out = LoadedImage{};
        return e;
    };
    if (base != h.image_base && !can_rebase) {
        return fail(make_error(ImageErrorCode::RelocationsStripped, 0, 0, base,
                               "preferred base unavailable and relocations stripped"));
    }

    // Commit (zeroed) every page a section touches, writable until payloads and relocations are
    // in place.
    const std::vector<PageRun> runs = page_protections(layout);
    uint64_t committed = 0;
    for (const PageRun& run : runs) {
        if (!arena.commit(base + run.rva, run.size, PageProtect::ReadWrite)) {
            return fail(make_error(ImageErrorCode::CommitFailed, kNoIndex, run.rva, run.size));
        }
        committed += run.size;
    }

    // Stream payloads one at a time straight into guest memory (the arena's backing is
    // contiguous), verifying each checksum as its bytes pass through.
    for (size_t i = 0; i < layout.sections.size(); ++i) {
        const SectionEntry& s = layout.sections[i];
        if (s.payload_length == 0) {
            continue;
        }
        uint64_t state = fnv1a_64(nullptr, 0);
        for (uint64_t done = 0; done < s.payload_length;) {
            const size_t n = static_cast<size_t>(
                std::min<uint64_t>(kStreamChunkBytes, s.payload_length - done));
            uint8_t* dst = arena.host_ptr(base + s.rva + done, n);
            if (dst == nullptr) {
                return fail(make_error(ImageErrorCode::WriteFailed, i, s.rva + done));
            }
            if (!reader.read(s.payload_offset + done, dst, n)) {
                return fail(make_error(ImageErrorCode::ReadFailed, i, s.payload_offset + done));
            }
            state = fnv1a_64(dst, n, state);
            done += n;
        }
        if (state != s.checksum) {
            return fail(make_error(ImageErrorCode::ChecksumMismatch, i, s.payload_offset, state));
        }
    }

    const bool rebased = base != h.image_base;
    uint64_t dir64_applied = 0;
    uint64_t highlow_applied = 0;
    if (rebased) {
        // Wrapping add: negative deltas work the same way as in apply_relocations (Rust).
        const uint64_t delta = base - h.image_base;
        for (uint32_t rva : layout.dir64) {
            uint8_t slot[8];
            if (!arena.read(base + rva, slot, 8)) {
                return fail(make_error(ImageErrorCode::WriteFailed, kNoIndex, rva));
            }
            put_le64(slot, le64(slot) + delta);
            if (!arena.write(base + rva, slot, 8)) {
                return fail(make_error(ImageErrorCode::WriteFailed, kNoIndex, rva));
            }
            ++dir64_applied;
        }
        for (uint32_t rva : layout.highlow) {
            uint8_t slot[4];
            if (!arena.read(base + rva, slot, 4)) {
                return fail(make_error(ImageErrorCode::WriteFailed, kNoIndex, rva));
            }
            put_le32(slot, le32(slot) + static_cast<uint32_t>(delta));
            if (!arena.write(base + rva, slot, 4)) {
                return fail(make_error(ImageErrorCode::WriteFailed, kNoIndex, rva));
            }
            ++highlow_applied;
        }
    }

    for (const PageRun& run : runs) {
        if (!arena.protect(base + run.rva, run.size, page_protect_for(run.protection))) {
            return fail(make_error(ImageErrorCode::ProtectFailed, kNoIndex, run.rva, run.size));
        }
    }

    LoadedImage result;
    result.base = base;
    result.preferred_base = h.image_base;
    result.size_of_image = h.size_of_image;
    result.entry_point = h.entry_point_rva != 0 ? base + h.entry_point_rva : 0;
    result.rebased = rebased;
    result.dir64_applied = dir64_applied;
    result.highlow_applied = highlow_applied;
    result.committed_bytes = committed;
    result.page_runs = runs;
    result.sections.reserve(layout.sections.size());
    for (const SectionEntry& s : layout.sections) {
        result.sections.push_back({s.name, base + s.rva, s.virtual_size, s.protection});
    }

    if (options.register_main_module != nullptr) {
        const ModuleRegistration& reg = *options.register_main_module;
        if (win32::register_main_module(reg.name, reg.path, base, h.size_of_image) == win32::kNullHandle) {
            return fail(make_error(ImageErrorCode::RegisterFailed, kNoIndex, 0, base));
        }
    }
    out = std::move(result);
    return {};
}

ImageError load_image(std::span<const uint8_t> bytes,
                      VirtualMemoryArena& arena,
                      const LoadOptions& options,
                      LoadedImage& out) {
    MemoryImageReader reader(bytes);
    return load_image(reader, arena, options, out);
}

bool unload_image(VirtualMemoryArena& arena, const LoadedImage& image) {
    return image.base != 0 && arena.release(image.base);
}

}  // namespace playgta6::os::pgim
