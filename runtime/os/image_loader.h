#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "os/vmem.h"

namespace playgta6::os::pgim {

// Loader for the PGIM v1 guest image container written by recomp/src/image.rs (that module is
// authoritative for the format). A container is validated (parse_layout / parse_image) and then
// mapped into a VirtualMemoryArena (load_image): reserve SizeOfImage, commit and zero the pages
// sections cover, copy payloads to base + RVA, rebase if base != preferred base, and finally set
// page protections.

inline constexpr uint32_t kVersion = 1;
inline constexpr uint64_t kHeaderBytes = 64;
inline constexpr uint64_t kSectionEntryBytes = 64;
inline constexpr uint64_t kPayloadAlignment = 4096;  // == kPageSize
inline constexpr size_t kSectionNameBytes = 16;

inline constexpr uint32_t kFlagHeadersMapped = 1u << 0;
inline constexpr uint32_t kFlagExecutableBytesOmitted = 1u << 1;
inline constexpr uint32_t kFlagRelocsStripped = 1u << 2;

inline constexpr uint32_t kSectionFlagHeaders = 1u << 0;
inline constexpr uint32_t kSectionFlagBytesOmitted = 1u << 1;

inline constexpr uint32_t kProtRead = 1u << 0;
inline constexpr uint32_t kProtWrite = 1u << 1;
inline constexpr uint32_t kProtExecute = 1u << 2;

inline constexpr uint64_t kNoIndex = UINT64_MAX;

enum class ImageErrorCode : uint32_t {
    Ok = 0,
    // Validation (mirrors ImageError in recomp/src/image.rs).
    Truncated,               // offset = bytes needed, value = bytes available
    BadMagic,                // value = the 4 magic bytes, little-endian
    UnsupportedVersion,      // value = version
    BadHeaderField,          // detail = field name, value = offending value
    SectionOutOfImage,       // index = section, offset = RVA, value = in-memory size
    OverlappingSections,     // index = the later section
    InvalidSection,          // index = section, detail = reason
    PayloadOutOfRange,       // index = section, offset/value = payload offset/length
    PayloadMisaligned,       // index = section, offset = payload offset
    PayloadOverlap,          // index = the later section
    ChecksumMismatch,        // index = section, offset = payload offset, value = actual checksum
    RelocationsUnsorted,     // offset = RVA of the offending site
    OverlappingRelocations,  // offset = RVA of the later site
    RelocSiteOutOfRange,     // offset = RVA of the site
    RelocSiteUnmapped,       // offset = RVA of the site
    // Reader and loader failures.
    ReadFailed,              // offset = container offset of the failed read
    InvalidArgument,
    RelocationsStripped,     // image cannot be rebased but the base differs from the preferred one
    ReserveFailed,
    CommitFailed,            // index = section, or kNoIndex when not tied to one
    WriteFailed,             // index = section, or kNoIndex for relocation sites (offset = RVA)
    ProtectFailed,
    RegisterFailed,          // register_main_module refused the module
};

struct ImageError {
    ImageErrorCode code = ImageErrorCode::Ok;
    uint64_t index = 0;   // section index where applicable
    uint64_t offset = 0;  // container offset, or RVA for section/relocation errors
    uint64_t value = 0;
    const char* detail = "";

    bool ok() const { return code == ImageErrorCode::Ok; }
};

const char* error_name(ImageErrorCode code);

// Random-access byte source. In the browser this is backed by HTTP range reads.
class ImageReader {
public:
    virtual ~ImageReader() = default;
    // Total container size in bytes.
    virtual uint64_t size() const = 0;
    // Fill dst with exactly len bytes at offset; false if the range cannot be served.
    virtual bool read(uint64_t offset, void* dst, size_t len) = 0;
};

class MemoryImageReader final : public ImageReader {
public:
    MemoryImageReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    explicit MemoryImageReader(std::span<const uint8_t> bytes)
        : data_(bytes.data()), size_(bytes.size()) {}
    uint64_t size() const override { return size_; }
    bool read(uint64_t offset, void* dst, size_t len) override;

private:
    const uint8_t* data_;
    size_t size_;
};

// Adapter for a "read range" callback (offset, destination, length -> success).
class FunctionImageReader final : public ImageReader {
public:
    using ReadFn = std::function<bool(uint64_t offset, void* dst, size_t len)>;
    FunctionImageReader(uint64_t size, ReadFn fn) : size_(size), fn_(std::move(fn)) {}
    uint64_t size() const override { return size_; }
    bool read(uint64_t offset, void* dst, size_t len) override;

private:
    uint64_t size_;
    ReadFn fn_;
};

struct ImageHeader {
    uint32_t flags = 0;
    uint64_t image_base = 0;
    uint32_t size_of_image = 0;
    uint32_t section_alignment = 0;
    uint32_t entry_point_rva = 0;
    uint32_t section_count = 0;
    uint32_t dir64_count = 0;
    uint32_t highlow_count = 0;
    uint64_t file_size = 0;

    bool headers_mapped() const { return (flags & kFlagHeadersMapped) != 0; }
    bool executable_bytes_omitted() const { return (flags & kFlagExecutableBytesOmitted) != 0; }
    bool relocs_stripped() const { return (flags & kFlagRelocsStripped) != 0; }
};

struct SectionEntry {
    std::string name;
    uint32_t rva = 0;
    uint32_t virtual_size = 0;
    uint32_t protection = 0;  // kProt* bits
    uint32_t flags = 0;       // kSectionFlag* bits
    uint64_t payload_offset = 0;
    uint64_t payload_length = 0;
    uint64_t checksum = 0;  // FNV-1a 64 of the payload
    uint32_t characteristics = 0;

    bool is_headers() const { return (flags & kSectionFlagHeaders) != 0; }
    bool bytes_omitted() const { return (flags & kSectionFlagBytesOmitted) != 0; }
    uint64_t end() const { return static_cast<uint64_t>(rva) + virtual_size; }
};

// Header, section table and relocation table of a structurally validated container. Everything
// parse_layout checks avoids the payload bytes; checksums are verified when payloads are read
// (parse_image does it for all payloads, load_image while streaming).
struct ImageLayout {
    ImageHeader header;
    std::vector<SectionEntry> sections;  // sorted by RVA, non-overlapping
    std::vector<uint32_t> dir64;         // strictly ascending site RVAs
    std::vector<uint32_t> highlow;       // strictly ascending site RVAs
};

// Fully validated container held in memory; payload views borrow from the input bytes.
struct ParsedImage {
    ImageLayout layout;
    std::vector<std::span<const uint8_t>> payloads;  // one per section, empty if none stored
};

// Reads only the header and tables (never payloads) and performs every structural check of
// read_image in recomp/src/image.rs. It reads and allocates only what the container actually
// supplies, so hostile counts fail instead of exhausting memory.
ImageError parse_layout(ImageReader& reader, ImageLayout& out);

// parse_layout plus payload checksum verification, with zero-copy payload views into `bytes`.
ImageError parse_image(std::span<const uint8_t> bytes, ParsedImage& out);

// A run of guest pages sharing one protection (RVA and size are page aligned).
struct PageRun {
    uint64_t rva = 0;
    uint64_t size = 0;
    uint32_t protection = 0;  // kProt* bits
};

// Page-granular protections the loader applies (same as GuestImage::page_protections in Rust):
// pages shared by several sections get the union, untouched pages are not listed.
std::vector<PageRun> page_protections(const ImageLayout& layout);

// Closest PageProtect for kProt* bits (Protection::win32_page_protect in Rust). Win32 has no
// write-only protection, so write implies read.
PageProtect page_protect_for(uint32_t protection);

uint64_t fnv1a_64(const uint8_t* data, size_t len, uint64_t state = 0xcbf29ce484222325ULL);

struct ModuleRegistration {
    std::string_view name;  // e.g. "GTA6.exe"
    std::string_view path;  // may be empty
};

struct LoadOptions {
    // Base to reserve first; 0 means the image's preferred base.
    uint64_t requested_base = 0;
    // If the requested base is unavailable, let the arena choose one (needs relocations).
    bool allow_fallback = true;
    // If set, the loaded image is registered via win32::register_main_module.
    const ModuleRegistration* register_main_module = nullptr;
};

struct LoadedSection {
    std::string name;
    uint64_t va = 0;
    uint32_t virtual_size = 0;
    uint32_t protection = 0;  // kProt* bits
};

struct LoadedImage {
    uint64_t base = 0;
    uint64_t preferred_base = 0;
    uint64_t size_of_image = 0;
    uint64_t entry_point = 0;  // VA, 0 if the image has none
    bool rebased = false;
    uint64_t dir64_applied = 0;
    uint64_t highlow_applied = 0;
    uint64_t committed_bytes = 0;  // bytes committed for this image
    std::vector<LoadedSection> sections;
    std::vector<PageRun> page_runs;  // rva relative to base
};

// Maps the container into `arena`. Payloads are streamed one at a time from `reader` (header and
// tables first), each verified against its checksum as it is copied, so a bad payload is detected
// after earlier ones were written; on any failure everything reserved is released again.
ImageError load_image(ImageReader& reader,
                      VirtualMemoryArena& arena,
                      const LoadOptions& options,
                      LoadedImage& out);

// Convenience for a fully in-memory container.
ImageError load_image(std::span<const uint8_t> bytes,
                      VirtualMemoryArena& arena,
                      const LoadOptions& options,
                      LoadedImage& out);

// Releases a loaded image's reservation.
bool unload_image(VirtualMemoryArena& arena, const LoadedImage& image);

}  // namespace playgta6::os::pgim
