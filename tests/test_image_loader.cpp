#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "os/image_loader.h"
#include "os/vmem.h"
#include "os/win32_loader.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_image_loader.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::pgim;

constexpr uint64_t kAny = UINT64_MAX - 1;  // "do not check the index"

void check_code_impl(const ImageError& err, ImageErrorCode code, uint64_t index, int line) {
    if (err.code != code || (index != kAny && err.index != index)) {
        std::fprintf(stderr,
                     "FAIL [test_image_loader.cpp:%d]: got %s (index %llu, offset 0x%llx, %s), "
                     "expected %s (index %llu)\n",
                     line, error_name(err.code), static_cast<unsigned long long>(err.index),
                     static_cast<unsigned long long>(err.offset), err.detail, error_name(code),
                     static_cast<unsigned long long>(index));
        ++g_failures;
    }
}

#define CHECK_CODE(err, code) check_code_impl((err), (code), kAny, __LINE__)
#define CHECK_CODE_AT(err, code, index) check_code_impl((err), (code), (index), __LINE__)

// ---------------------------------------------------------------------------
// Synthetic containers: a small writer mirroring write_image in recomp/src/image.rs.
// ---------------------------------------------------------------------------

constexpr uint64_t kBase = 0x2'0000'0000ULL;

struct Sec {
    std::string name;
    uint32_t rva = 0;
    uint32_t vsize = 0;
    uint32_t prot = 0;
    uint32_t flags = 0;
    std::vector<uint8_t> data;
    uint32_t chars = 0;
};

struct Img {
    uint64_t base = kBase;
    uint32_t size = 0;
    uint32_t align = 0x1000;
    uint32_t entry = 0;
    uint32_t flags = 0;
    std::vector<Sec> secs;
    std::vector<uint32_t> dir64;
    std::vector<uint32_t> highlow;
};

void put32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void patch32(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

void patch64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

uint32_t get32(const std::vector<uint8_t>& b, size_t at) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | b[at + i];
    return v;
}

uint64_t get64(const std::vector<uint8_t>& b, size_t at) {
    return static_cast<uint64_t>(get32(b, at)) | static_cast<uint64_t>(get32(b, at + 4)) << 32;
}

uint64_t align_page(uint64_t v) { return (v + 4095) / 4096 * 4096; }

std::vector<uint8_t> write_container(const Img& img) {
    const uint64_t tables =
        64 + 64 * img.secs.size() + 4 * (img.dir64.size() + img.highlow.size());
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables;
    for (const Sec& s : img.secs) {
        if (s.data.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = align_page(cursor);
        offsets.push_back(off);
        cursor = off + s.data.size();
    }
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put32(out, 1);
    put32(out, 64);
    put32(out, 64);
    put32(out, img.flags);
    put32(out, 4096);
    put64(out, img.base);
    put32(out, img.size);
    put32(out, img.align);
    put32(out, img.entry);
    put32(out, static_cast<uint32_t>(img.secs.size()));
    put32(out, static_cast<uint32_t>(img.dir64.size()));
    put32(out, static_cast<uint32_t>(img.highlow.size()));
    put64(out, cursor);
    for (size_t i = 0; i < img.secs.size(); ++i) {
        const Sec& s = img.secs[i];
        std::string name = s.name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put32(out, s.rva);
        put32(out, s.vsize);
        put32(out, s.prot);
        put32(out, s.flags);
        put64(out, offsets[i]);
        put64(out, s.data.size());
        put64(out, fnv1a_64(s.data.data(), s.data.size()));
        put32(out, s.chars);
        put32(out, 0);
    }
    for (uint32_t r : img.dir64) put32(out, r);
    for (uint32_t r : img.highlow) put32(out, r);
    for (size_t i = 0; i < img.secs.size(); ++i) {
        if (!img.secs[i].data.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), img.secs[i].data.begin(), img.secs[i].data.end());
        }
    }
    return out;
}

std::vector<uint8_t> pattern(uint32_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = static_cast<uint8_t>((x >> 24) | 1u);  // never zero, so zero-fill is visible
    }
    return v;
}

constexpr uint32_t R = kProtRead;
constexpr uint32_t W = kProtWrite;
constexpr uint32_t X = kProtExecute;

// Layout (page numbers in brackets):
//   <headers> 0x0000..0x0400 R [0]        .text  0x1000..0x3000 RX [1,2]
//   .data     0x3000..0x3800 RW [3]       .rdata 0x3800..0x3C00 R  [3, shares the page]
//   (page 4 unmapped)                     .bss   0x5000..0x7800 RW [5,6,7]
//   .tail     0x7C00..0x7E00 RX [7, shares with .bss -> RWX]        (page 8 unmapped)
Img fixture() {
    Img img;
    img.size = 0x9000;
    img.entry = 0x1010;
    img.flags = kFlagHeadersMapped;
    img.secs = {
        {"<headers>", 0, 0x400, R, kSectionFlagHeaders, pattern(1, 0x180), 0},
        {".text", 0x1000, 0x2000, R | X, 0, pattern(2, 0x1800), 0x60000020},
        {".data", 0x3000, 0x800, R | W, 0, pattern(3, 0x7F0), 0xC0000040},
        {".rdata", 0x3800, 0x400, R, 0, pattern(4, 0x400), 0x40000040},
        {".bss", 0x5000, 0x2800, R | W, 0, {}, 0xC0000080},
        {".tail", 0x7C00, 0x200, R | X, 0, pattern(5, 0x100), 0x60000020},
    };
    img.dir64 = {0x1008, 0x2FFC, 0x3011, 0x5008};  // 0x2FFC straddles a page and two sections
    img.highlow = {0x1100, 0x1FFE, 0x3100, 0x7C10};  // 0x1FFE straddles a page
    // Make the site values interesting: carries and wrap-around.
    auto set64 = [&](size_t sec, uint32_t rva, uint64_t v) {
        for (int i = 0; i < 8; ++i) img.secs[sec].data[rva - img.secs[sec].rva + i] = uint8_t(v >> (8 * i));
    };
    set64(1, 0x1008, 0x0000'0002'0000'1234ULL);
    set64(2, 0x3011, 0xFFFF'FFFF'FFFF'FFF0ULL);
    return img;
}

// Flat guest memory [0, size) at the preferred base, as the Rust `materialize` helper.
std::vector<uint8_t> flat_memory(const Img& img) {
    std::vector<uint8_t> mem(img.size, 0);
    for (const Sec& s : img.secs) {
        std::copy(s.data.begin(), s.data.end(), mem.begin() + s.rva);
    }
    return mem;
}

// Reference relocation (apply_relocations in Rust).
void rebase_flat(std::vector<uint8_t>& mem, const Img& img, uint64_t delta) {
    for (uint32_t rva : img.dir64) {
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | mem[rva + i];
        v += delta;
        for (int i = 0; i < 8; ++i) mem[rva + i] = uint8_t(v >> (8 * i));
    }
    for (uint32_t rva : img.highlow) {
        uint32_t v = 0;
        for (int i = 3; i >= 0; --i) v = (v << 8) | mem[rva + i];
        v += static_cast<uint32_t>(delta);
        for (int i = 0; i < 4; ++i) mem[rva + i] = uint8_t(v >> (8 * i));
    }
}

// Independent oracle for page protection bits -> PageProtect.
PageProtect oracle_protect(uint32_t bits) {
    switch (bits & 7u) {
        case 0: return PageProtect::NoAccess;
        case 1: return PageProtect::ReadOnly;
        case 2:
        case 3: return PageProtect::ReadWrite;
        case 4:
        case 5: return PageProtect::ExecuteRead;
        default: return PageProtect::ExecuteReadWrite;
    }
}

// Every byte at its RVA (relocated by base - preferred), zero-fill zero, per-page state and
// protection (union over sections sharing a page), untouched pages still only reserved.
void verify_memory(VirtualMemoryArena& arena, const Img& img, const LoadedImage& li, int line) {
    const uint64_t base = li.base;
    std::vector<uint8_t> expect = flat_memory(img);
    rebase_flat(expect, img, base - img.base);
    bool ok = true;
    for (uint64_t p = 0; p < img.size / 4096; ++p) {
        const uint64_t lo = p * 4096;
        const uint64_t hi = lo + 4096;
        bool touched = false;
        uint32_t bits = 0;
        for (const Sec& s : img.secs) {
            if (s.rva < hi && uint64_t{s.rva} + s.vsize > lo) {
                touched = true;
                bits |= s.prot;
            }
        }
        const MemoryBasicInfo info = arena.query(base + lo);
        if (!touched) {
            ok &= info.state == MemState::Reserve && info.protect == PageProtect::NoAccess;
            uint8_t b;
            ok &= !arena.read(base + lo, &b, 1);
            continue;
        }
        ok &= info.state == MemState::Commit;
        ok &= info.protect == oracle_protect(bits);
        if (is_readable(info.protect)) {
            std::vector<uint8_t> page(4096);
            ok &= arena.read(base + lo, page.data(), 4096);
            ok &= std::memcmp(page.data(), expect.data() + lo, 4096) == 0;
        }
        ok &= is_writable(info.protect) == ((bits & W) != 0);
        ok &= is_executable(info.protect) == ((bits & X) != 0);
    }
    check_impl(ok, "verify_memory", line);
}

#define VERIFY_MEMORY(arena, img, li) verify_memory((arena), (img), (li), __LINE__)

struct Baseline {
    uint64_t reserved;
    uint64_t committed;
    size_t count;
};

Baseline baseline_of(const VirtualMemoryArena& a) {
    return {a.reserved_bytes(), a.committed_bytes(), a.reservation_count()};
}

void check_baseline_impl(const VirtualMemoryArena& a, const Baseline& b, int line) {
    check_impl(a.reserved_bytes() == b.reserved && a.committed_bytes() == b.committed &&
                   a.reservation_count() == b.count,
               "arena accounting back to baseline", line);
}

#define CHECK_BASELINE(arena, b) check_baseline_impl((arena), (b), __LINE__)

ImageError parse_err(const std::vector<uint8_t>& bytes) {
    ParsedImage parsed;
    return parse_image(bytes, parsed);
}

#define EXPECT_PARSE(bytes, code) check_code_impl(parse_err(bytes), (code), kAny, __LINE__)
#define EXPECT_PARSE_AT(bytes, code, index) check_code_impl(parse_err(bytes), (code), (index), __LINE__)

ImageError load(VirtualMemoryArena& arena,
                const std::vector<uint8_t>& bytes,
                LoadedImage& out,
                uint64_t requested = 0,
                bool fallback = true) {
    LoadOptions opts;
    opts.requested_base = requested;
    opts.allow_fallback = fallback;
    return load_image(bytes, arena, opts, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_primitives() {
    CHECK(fnv1a_64(nullptr, 0) == 0xcbf29ce484222325ULL);
    const uint8_t a = 'a';
    CHECK(fnv1a_64(&a, 1) == 0xaf63dc4c8601ec8cULL);
    const uint8_t abc[] = {'a', 'b', 'c'};
    CHECK(fnv1a_64(abc, 3) == fnv1a_64(abc + 1, 2, fnv1a_64(abc, 1)));

    CHECK(page_protect_for(0) == PageProtect::NoAccess);
    CHECK(page_protect_for(R) == PageProtect::ReadOnly);
    CHECK(page_protect_for(W) == PageProtect::ReadWrite);
    CHECK(page_protect_for(R | W) == PageProtect::ReadWrite);
    CHECK(page_protect_for(X) == PageProtect::ExecuteRead);
    CHECK(page_protect_for(R | X) == PageProtect::ExecuteRead);
    CHECK(page_protect_for(W | X) == PageProtect::ExecuteReadWrite);
    CHECK(page_protect_for(R | W | X) == PageProtect::ExecuteReadWrite);
    // The raw values are the Win32 PAGE_* constants the Rust side emits.
    CHECK(static_cast<uint32_t>(page_protect_for(R | W | X)) == 0x40);
    CHECK(static_cast<uint32_t>(page_protect_for(R | X)) == 0x20);
    CHECK(static_cast<uint32_t>(page_protect_for(W)) == 0x04);
    CHECK(static_cast<uint32_t>(page_protect_for(R)) == 0x02);
    CHECK(static_cast<uint32_t>(page_protect_for(0)) == 0x01);
}

void test_parse_image_views() {
    const Img img = fixture();
    const std::vector<uint8_t> bytes = write_container(img);
    ParsedImage parsed;
    CHECK(parse_image(bytes, parsed).ok());
    const ImageHeader& h = parsed.layout.header;
    CHECK(h.image_base == kBase && h.size_of_image == 0x9000 && h.entry_point_rva == 0x1010);
    CHECK(h.headers_mapped() && !h.executable_bytes_omitted() && !h.relocs_stripped());
    CHECK(h.section_count == 6 && h.dir64_count == 4 && h.highlow_count == 4);
    CHECK(h.file_size == bytes.size());
    CHECK(parsed.layout.sections.size() == 6 && parsed.payloads.size() == 6);
    CHECK(parsed.layout.dir64 == img.dir64 && parsed.layout.highlow == img.highlow);
    for (size_t i = 0; i < img.secs.size(); ++i) {
        const SectionEntry& s = parsed.layout.sections[i];
        CHECK(s.name == img.secs[i].name && s.rva == img.secs[i].rva);
        CHECK(s.virtual_size == img.secs[i].vsize && s.protection == img.secs[i].prot);
        CHECK(s.characteristics == img.secs[i].chars);
        CHECK(s.payload_offset % 4096 == 0);
        const auto view = parsed.payloads[i];
        CHECK(view.size() == img.secs[i].data.size());
        if (!view.empty()) {
            // Zero-copy: the view points into the input buffer.
            CHECK(view.data() >= bytes.data() && view.data() + view.size() <= bytes.data() + bytes.size());
            CHECK(std::memcmp(view.data(), img.secs[i].data.data(), view.size()) == 0);
        } else {
            CHECK(s.payload_offset == 0);
        }
    }
    CHECK(parsed.layout.sections[0].is_headers() && !parsed.layout.sections[1].is_headers());

    const std::vector<PageRun> runs = page_protections(parsed.layout);
    const PageRun expected[] = {
        {0x0000, 0x1000, R},         {0x1000, 0x2000, R | X},
        {0x3000, 0x1000, R | W},     {0x5000, 0x2000, R | W},
        {0x7000, 0x1000, R | W | X},  // .bss + .tail share the page: union
    };
    CHECK(runs.size() == 5);
    for (size_t i = 0; i < runs.size() && i < 5; ++i) {
        CHECK(runs[i].rva == expected[i].rva && runs[i].size == expected[i].size &&
              runs[i].protection == expected[i].protection);
    }
}

void test_roundtrip_preferred_base() {
    const Img img = fixture();
    const std::vector<uint8_t> bytes = write_container(img);
    VirtualMemoryArena arena;
    const Baseline base0 = baseline_of(arena);

    LoadedImage li;
    CHECK(load(arena, bytes, li).ok());
    CHECK(li.base == kBase && li.preferred_base == kBase && !li.rebased);
    CHECK(li.dir64_applied == 0 && li.highlow_applied == 0);
    CHECK(li.size_of_image == 0x9000 && li.entry_point == kBase + 0x1010);
    CHECK(li.sections.size() == 6);
    for (size_t i = 0; i < li.sections.size() && i < 6; ++i) {
        CHECK(li.sections[i].va == kBase + img.secs[i].rva);
        CHECK(li.sections[i].name == img.secs[i].name);
        CHECK(li.sections[i].virtual_size == img.secs[i].vsize);
        CHECK(li.sections[i].protection == img.secs[i].prot);
    }
    CHECK(li.page_runs.size() == 5);
    CHECK(li.committed_bytes == 0x1000 + 0x2000 + 0x1000 + 0x2000 + 0x1000);
    CHECK(arena.reserved_bytes() == 0x9000);
    CHECK(arena.committed_bytes() == li.committed_bytes);
    CHECK(arena.reservation_count() == 1);
    VERIFY_MEMORY(arena, img, li);

    // Page 4 and page 8 stay reserved and inaccessible.
    CHECK(arena.query(kBase + 0x4000).state == MemState::Reserve);
    CHECK(arena.query(kBase + 0x8000).state == MemState::Reserve);
    // Protections are enforced by the arena: .text is not writable, .data is.
    const uint8_t zero = 0;
    CHECK(!arena.write(kBase + 0x1000, &zero, 1));
    CHECK(arena.write(kBase + 0x3000, &zero, 1));

    CHECK(unload_image(arena, li));
    CHECK_BASELINE(arena, base0);
    CHECK(!unload_image(arena, li));
}

void test_rebase() {
    const Img img = fixture();
    const std::vector<uint8_t> bytes = write_container(img);
    const struct {
        uint64_t base;
        uint64_t dir64;
    } cases[] = {
        {0x2'8000'0000ULL, 4},   // positive delta 0x8000'0000
        {0x1'8000'0000ULL, 4},   // negative delta (64-bit wrap, low 32 bits wrap too)
        {0x3'8000'0000ULL, 4},   // delta with high bits set: HIGHLOW adds only the low 32 bits
        {0x2'0001'0000ULL, 4},   // small positive delta
    };
    for (const auto& c : cases) {
        VirtualMemoryArena arena;
        const Baseline base0 = baseline_of(arena);
        LoadedImage li;
        CHECK(load(arena, bytes, li, c.base).ok());
        CHECK(li.base == c.base && li.rebased && li.preferred_base == kBase);
        CHECK(li.dir64_applied == c.dir64 && li.highlow_applied == 4);
        CHECK(li.entry_point == c.base + 0x1010);
        VERIFY_MEMORY(arena, img, li);

        // Spot-check the interesting sites by hand: the unaligned DIR64 site at 0x3011 held
        // 0xFFFF'FFFF'FFFF'FFF0 and wraps; 0x2FFC straddles the page boundary at 0x3000.
        const uint64_t delta = c.base - kBase;
        uint64_t v = 0;
        CHECK(arena.read(c.base + 0x1008, &v, 8));
        CHECK(v == 0x0000'0002'0000'1234ULL + delta);
        CHECK(arena.read(c.base + 0x3011, &v, 8));
        CHECK(v == 0xFFFF'FFFF'FFFF'FFF0ULL + delta);
        std::vector<uint8_t> exp = flat_memory(img);
        rebase_flat(exp, img, delta);
        uint8_t straddle[8];
        CHECK(arena.read(c.base + 0x2FFC, straddle, 8));
        CHECK(std::memcmp(straddle, exp.data() + 0x2FFC, 8) == 0);
        uint8_t hl[4];
        CHECK(arena.read(c.base + 0x1FFE, hl, 4));
        CHECK(std::memcmp(hl, exp.data() + 0x1FFE, 4) == 0);

        CHECK(unload_image(arena, li));
        CHECK_BASELINE(arena, base0);
    }

    // Preferred base occupied: the arena picks another base and the image is rebased.
    {
        VirtualMemoryArena arena;
        CHECK(arena.reserve(kBase, 0x10000, PageProtect::ReadWrite) == kBase);
        const Baseline base0 = baseline_of(arena);
        LoadedImage li;
        CHECK(load(arena, bytes, li).ok());
        CHECK(li.base != kBase && li.rebased);
        VERIFY_MEMORY(arena, img, li);
        CHECK(unload_image(arena, li));
        CHECK_BASELINE(arena, base0);

        CHECK_CODE(load(arena, bytes, li, 0, false), ImageErrorCode::ReserveFailed);
        CHECK_BASELINE(arena, base0);
    }
}

void test_relocations_stripped() {
    Img img = fixture();
    img.flags |= kFlagRelocsStripped;
    img.dir64.clear();
    img.highlow.clear();
    const std::vector<uint8_t> bytes = write_container(img);

    {
        VirtualMemoryArena arena;
        LoadedImage li;
        CHECK(load(arena, bytes, li).ok());
        CHECK(li.base == kBase && !li.rebased);
        VERIFY_MEMORY(arena, img, li);
    }
    {
        VirtualMemoryArena arena;
        const Baseline base0 = baseline_of(arena);
        LoadedImage li;
        CHECK_CODE(load(arena, bytes, li, 0x2'8000'0000ULL), ImageErrorCode::RelocationsStripped);
        CHECK_BASELINE(arena, base0);
    }
    {
        // Preferred base taken: there is no fallback for an image that cannot be rebased.
        VirtualMemoryArena arena;
        CHECK(arena.reserve(kBase, 0x10000, PageProtect::ReadWrite) == kBase);
        const Baseline base0 = baseline_of(arena);
        LoadedImage li;
        CHECK_CODE(load(arena, bytes, li), ImageErrorCode::ReserveFailed);
        CHECK_BASELINE(arena, base0);
    }
}

void test_omitted_executable_bytes() {
    Img img;
    img.size = 0x5000;
    img.flags = kFlagHeadersMapped | kFlagExecutableBytesOmitted;
    img.secs = {
        {"<headers>", 0, 0x200, R, kSectionFlagHeaders, pattern(1, 0x100), 0},
        {".text", 0x1000, 0x2000, R | X, kSectionFlagBytesOmitted, {}, 0x60000020},
        {".data", 0x3000, 0x1000, R | W, 0, pattern(2, 0x1000), 0xC0000040},
    };
    img.dir64 = {0x3000};  // relocation sites only in sections whose bytes are stored
    img.highlow = {0x3FFC};
    const std::vector<uint8_t> bytes = write_container(img);

    ParsedImage parsed;
    CHECK(parse_image(bytes, parsed).ok());
    CHECK(parsed.layout.header.executable_bytes_omitted());
    CHECK(parsed.layout.sections[1].bytes_omitted() && parsed.payloads[1].empty());

    VirtualMemoryArena arena;
    LoadedImage li;
    CHECK(load(arena, bytes, li, 0x2'0010'0000ULL).ok());
    CHECK(li.rebased && li.dir64_applied == 1 && li.highlow_applied == 1);
    VERIFY_MEMORY(arena, img, li);  // .text is zero-fill with its original protection
    CHECK(arena.query(li.base + 0x1000).protect == PageProtect::ExecuteRead);
    CHECK(arena.query(li.base + 0x2000).protect == PageProtect::ExecuteRead);
    CHECK(arena.query(li.base).protect == PageProtect::ReadOnly);  // headers pseudo-section
    uint8_t magic = 0;
    CHECK(arena.read(li.base, &magic, 1) && magic == img.secs[0].data[0]);

    // A relocation site inside an omitted section is unmapped.
    Img bad = img;
    bad.dir64 = {0x1000};
    EXPECT_PARSE(write_container(bad), ImageErrorCode::RelocSiteUnmapped);
    // Executable bytes must be omitted when the flag says so.
    bad = img;
    bad.secs[1].flags = 0;
    EXPECT_PARSE_AT(write_container(bad), ImageErrorCode::InvalidSection, 1);
    // Omitted flag on a section that stores bytes, or without the header flag.
    bad = img;
    bad.secs[2].flags = kSectionFlagBytesOmitted;
    EXPECT_PARSE_AT(write_container(bad), ImageErrorCode::InvalidSection, 2);
    bad = img;
    bad.flags = kFlagHeadersMapped;
    EXPECT_PARSE_AT(write_container(bad), ImageErrorCode::InvalidSection, 1);
}

void test_protection_variants() {
    Img img;
    img.size = 0x8000;
    img.secs = {
        {"none", 0x0000, 0x100, 0, 0, pattern(1, 0x80), 0},
        {"wo", 0x1000, 0x100, W, 0, pattern(2, 0x80), 0},
        {"xo", 0x2000, 0x100, X, 0, pattern(3, 0x80), 0},
        {"wx", 0x3000, 0x100, W | X, 0, pattern(4, 0x80), 0},
        {"rwx", 0x4000, 0x100, R | W | X, 0, pattern(5, 0x80), 0},
        // A section that ends mid-page followed by one that starts later in the same page.
        {"a", 0x5000, 0x10, R, 0, pattern(6, 0x10), 0},
        {"b", 0x5800, 0x10, W | X, 0, pattern(7, 0x10), 0},
    };
    const std::vector<uint8_t> bytes = write_container(img);
    VirtualMemoryArena arena;
    LoadedImage li;
    CHECK(load(arena, bytes, li).ok());
    VERIFY_MEMORY(arena, img, li);
    CHECK(arena.query(kBase).state == MemState::Commit);
    CHECK(arena.query(kBase).protect == PageProtect::NoAccess);
    CHECK(arena.query(kBase + 0x1000).protect == PageProtect::ReadWrite);
    CHECK(arena.query(kBase + 0x2000).protect == PageProtect::ExecuteRead);
    CHECK(arena.query(kBase + 0x3000).protect == PageProtect::ExecuteReadWrite);
    CHECK(arena.query(kBase + 0x4000).protect == PageProtect::ExecuteReadWrite);
    CHECK(arena.query(kBase + 0x5000).protect == PageProtect::ExecuteReadWrite);  // R | WX
    CHECK(arena.query(kBase + 0x6000).state == MemState::Reserve);
    uint8_t b;
    CHECK(!arena.read(kBase, &b, 1));  // no access
}

void test_corrupt_containers() {
    const Img img = fixture();
    const std::vector<uint8_t> good = write_container(img);
    CHECK(parse_err(good).ok());
    auto entry = [](size_t i) { return size_t{64} + 64 * i; };
    const size_t relocs = 64 + 64 * 6;  // 4 DIR64 then 4 HIGHLOW
    auto with32 = [&](size_t at, uint32_t v) {
        std::vector<uint8_t> b = good;
        patch32(b, at, v);
        return b;
    };
    auto with64 = [&](size_t at, uint64_t v) {
        std::vector<uint8_t> b = good;
        patch64(b, at, v);
        return b;
    };

    // Header.
    EXPECT_PARSE(std::vector<uint8_t>(good.begin(), good.begin() + 63), ImageErrorCode::Truncated);
    EXPECT_PARSE(std::vector<uint8_t>(), ImageErrorCode::Truncated);
    {
        std::vector<uint8_t> b = good;
        b[0] = 'M';
        b[1] = 'Z';
        EXPECT_PARSE(b, ImageErrorCode::BadMagic);
    }
    check_code_impl(parse_err(with32(4, 2)), ImageErrorCode::UnsupportedVersion, kAny, __LINE__);
    EXPECT_PARSE(with32(8, 80), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(with32(12, 32), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(with32(20, 8192), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(with32(16, kFlagHeadersMapped | 0x80), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(with64(56, good.size() + 1), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(std::vector<uint8_t>(good.begin(), good.end() - 1), ImageErrorCode::BadHeaderField);
    EXPECT_PARSE(with32(44, 0x00FFFFFF), ImageErrorCode::Truncated);
    EXPECT_PARSE(with32(48, 0xFFFFFFFF), ImageErrorCode::Truncated);
    EXPECT_PARSE(with32(52, 0xFFFFFFFF), ImageErrorCode::Truncated);
    EXPECT_PARSE(with64(24, UINT64_MAX - 0x100), ImageErrorCode::BadHeaderField);  // base + size wraps
    CHECK(std::strcmp(parse_err(with32(20, 8192)).detail, "page_size") == 0);
    CHECK(std::strcmp(parse_err(with32(16, 0x80)).detail, "flags") == 0);

    // Section table: ordering, bounds, fields.
    EXPECT_PARSE_AT(with32(entry(2) + 16, 0x1800), ImageErrorCode::OverlappingSections, 2);  // .data on .text
    EXPECT_PARSE_AT(with32(entry(1) + 16, 0x6000), ImageErrorCode::OverlappingSections, 2);  // unsorted
    EXPECT_PARSE_AT(with32(entry(5) + 20, 0x9000), ImageErrorCode::SectionOutOfImage, 5);
    EXPECT_PARSE_AT(with32(entry(5) + 16, 0xFFFFFF00), ImageErrorCode::SectionOutOfImage, 5);  // u32 sum
    EXPECT_PARSE_AT(with32(entry(2) + 20, 0x10), ImageErrorCode::InvalidSection, 2);  // bytes > size
    EXPECT_PARSE_AT(with32(entry(4) + 20, 0), ImageErrorCode::InvalidSection, 4);     // zero size
    EXPECT_PARSE_AT(with32(entry(2) + 24, 8), ImageErrorCode::InvalidSection, 2);     // protection
    EXPECT_PARSE_AT(with32(entry(2) + 28, 4), ImageErrorCode::InvalidSection, 2);     // flags
    EXPECT_PARSE_AT(with32(entry(2) + 60, 1), ImageErrorCode::InvalidSection, 2);     // reserved
    EXPECT_PARSE_AT(with32(entry(2) + 28, kSectionFlagHeaders), ImageErrorCode::InvalidSection, 2);
    EXPECT_PARSE_AT(with32(16, 0), ImageErrorCode::InvalidSection, 0);  // headers flag vs section
    {
        std::vector<uint8_t> b = good;
        b[entry(1) + 10] = 'x';  // junk after the NUL terminator
        EXPECT_PARSE_AT(b, ImageErrorCode::InvalidSection, 1);
        b = good;
        b[entry(1)] = 0xFF;  // not UTF-8
        EXPECT_PARSE_AT(b, ImageErrorCode::InvalidSection, 1);
        b = good;
        b[entry(1)] = 0xC0;  // overlong lead byte
        b[entry(1) + 1] = 0x80;
        EXPECT_PARSE_AT(b, ImageErrorCode::InvalidSection, 1);
        b = good;
        b[entry(1)] = 0xED;  // surrogate
        b[entry(1) + 1] = 0xA0;
        b[entry(1) + 2] = 0x80;
        EXPECT_PARSE_AT(b, ImageErrorCode::InvalidSection, 1);
        b = good;
        // A valid 16-byte name without a terminator is fine.
        std::memset(&b[entry(1)], 'n', 16);
        CHECK(parse_err(b).ok());
        b = good;
        std::memset(&b[entry(1)], 0, 16);
        b[entry(1)] = 0xC3;  // valid two-byte sequence
        b[entry(1) + 1] = 0xA9;
        CHECK(parse_err(b).ok());
    }

    // Payloads.
    const uint64_t data_off = get64(good, entry(2) + 32);
    EXPECT_PARSE_AT(with64(entry(2) + 32, uint64_t{1} << 40), ImageErrorCode::PayloadOutOfRange, 2);
    EXPECT_PARSE_AT(with64(entry(2) + 40, UINT64_MAX), ImageErrorCode::PayloadOutOfRange, 2);
    EXPECT_PARSE_AT(with64(entry(2) + 32, data_off + 1), ImageErrorCode::PayloadMisaligned, 2);
    EXPECT_PARSE_AT(with64(entry(2) + 32, 0), ImageErrorCode::PayloadOutOfRange, 2);  // inside tables
    EXPECT_PARSE_AT(with64(entry(2) + 32, 4096), ImageErrorCode::PayloadOverlap, kAny);  // .headers' page
    EXPECT_PARSE_AT(with64(entry(4) + 32, 4096), ImageErrorCode::InvalidSection, 4);  // empty + offset
    EXPECT_PARSE_AT(with64(entry(4) + 48, 1), ImageErrorCode::ChecksumMismatch, 4);   // empty checksum
    {
        std::vector<uint8_t> aliased = good;  // .text takes .data's offset, length and checksum
        std::memcpy(&aliased[entry(1) + 32], &good[entry(2) + 32], 24);
        EXPECT_PARSE_AT(aliased, ImageErrorCode::PayloadOverlap, kAny);
        std::vector<uint8_t> flipped = good;
        flipped[static_cast<size_t>(data_off) + 8] ^= 0xFF;
        EXPECT_PARSE_AT(flipped, ImageErrorCode::ChecksumMismatch, 2);
        flipped = good;
        flipped.back() ^= 1;  // last byte of the last payload
        EXPECT_PARSE_AT(flipped, ImageErrorCode::ChecksumMismatch, 5);
    }

    // Relocation lists: DIR64 [0x1008, 0x2FFC, 0x3011, 0x5008], HIGHLOW [0x1100, 0x1FFE, 0x3100, 0x7C10].
    const size_t hl = relocs + 16;
    EXPECT_PARSE(with32(relocs, 0x2FFD), ImageErrorCode::RelocationsUnsorted);
    EXPECT_PARSE(with32(relocs + 4, 0x1008), ImageErrorCode::RelocationsUnsorted);  // equal
    EXPECT_PARSE(with32(hl + 4, 0x1100), ImageErrorCode::RelocationsUnsorted);
    EXPECT_PARSE(with32(hl, 0x1008), ImageErrorCode::RelocationsUnsorted);  // same RVA in both lists
    EXPECT_PARSE(with32(relocs + 12, 0xFFFFFFF0), ImageErrorCode::RelocSiteOutOfRange);
    EXPECT_PARSE(with32(hl + 12, 0x8FFE), ImageErrorCode::RelocSiteOutOfRange);  // 4 bytes end past image
    {
        std::vector<uint8_t> b = good;
        patch32(b, relocs + 4, 0x1010);  // DIR64 [0x1008, 0x1010, ...]
        patch32(b, hl, 0x100C);          // HIGHLOW 0x100C inside the DIR64 at 0x1008
        const ImageError e = parse_err(b);
        CHECK_CODE(e, ImageErrorCode::OverlappingRelocations);
        CHECK(e.offset == 0x100C && e.value == 0x1008);
    }
    EXPECT_PARSE(with32(hl + 12, 0x4000), ImageErrorCode::RelocSiteUnmapped);  // page 4 is not mapped
    EXPECT_PARSE(with32(hl + 12, 0x7DFE), ImageErrorCode::RelocSiteUnmapped);  // runs off .tail
    EXPECT_PARSE(with32(hl + 12, 0x3BFE), ImageErrorCode::RelocSiteUnmapped);  // .rdata end -> gap
    {
        // Straddling two adjacent sections is fine: .data ends where .rdata starts.
        std::vector<uint8_t> b = good;
        patch32(b, hl + 8, 0x37FE);
        CHECK(parse_err(b).ok());
    }

    // Truncation at every table boundary and a few bytes beyond never validates.
    const size_t tables_end = relocs + 32;
    for (size_t len = 0; len < tables_end + 8; ++len) {
        const std::vector<uint8_t> cut(good.begin(), good.begin() + static_cast<long>(len));
        CHECK(!parse_err(cut).ok());
    }
    for (size_t at : {size_t{64}, size_t{128}, relocs, hl, tables_end, size_t{4096}, good.size() - 1}) {
        const std::vector<uint8_t> cut(good.begin(), good.begin() + static_cast<long>(at));
        CHECK(!parse_err(cut).ok());
    }
    // Single-byte corruptions of header and tables never crash the validator.
    for (size_t at = 0; at < tables_end; ++at) {
        for (uint8_t mask : {uint8_t{0x01}, uint8_t{0x80}, uint8_t{0xFF}}) {
            std::vector<uint8_t> b = good;
            b[at] ^= mask;
            ParsedImage parsed;
            const ImageError e = parse_image(b, parsed);
            if (e.ok()) {
                CHECK(parsed.payloads.size() == parsed.layout.sections.size());
            }
        }
    }
}

void test_cleanup_on_failure() {
    const Img img = fixture();
    const std::vector<uint8_t> good = write_container(img);
    VirtualMemoryArena arena;
    const uint64_t other = arena.reserve(0, 0x20000, PageProtect::ReadWrite);
    CHECK(other != 0);
    CHECK(arena.commit(other, 0x4000, PageProtect::ReadWrite));
    const Baseline base0 = baseline_of(arena);
    LoadedImage li;
    li.base = 123;

    // Validation failures reserve nothing.
    std::vector<uint8_t> bad = good;
    patch32(bad, 16, 0x80);
    CHECK_CODE(load(arena, bad, li), ImageErrorCode::BadHeaderField);
    CHECK(li.base == 0);
    CHECK_BASELINE(arena, base0);

    // A payload checksum is verified while streaming: earlier payloads were already copied, then
    // everything is released again. Hit each payload in turn (.text, .data, .rdata, .tail).
    for (size_t sec : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{5}}) {
        bad = good;
        const uint64_t off = get64(good, 64 + 64 * sec + 32);
        bad[static_cast<size_t>(off)] ^= 0x55;
        CHECK_CODE_AT(load(arena, bad, li), ImageErrorCode::ChecksumMismatch, sec);
        CHECK_BASELINE(arena, base0);
        CHECK(li.base == 0 && li.sections.empty());
    }

    // The reader failing mid-payload (e.g. a dropped HTTP range request).
    {
        MemoryImageReader mem(good);
        size_t calls = 0;
        FunctionImageReader failing(good.size(), [&](uint64_t off, void* dst, size_t len) {
            ++calls;
            if (off >= 4096 * 2) return false;  // everything from the second payload on
            return mem.read(off, dst, len);
        });
        LoadOptions opts;
        CHECK_CODE(load_image(failing, arena, opts, li), ImageErrorCode::ReadFailed);
        CHECK(calls > 3);
        CHECK_BASELINE(arena, base0);
    }

    // Registry conflicts after a successful mapping release the reservation again; the successful
    // case registers the main module (O7).
    {
        const ModuleRegistration reg{"pgim_loader_test.exe", "C:/pgim_loader_test.exe"};
        LoadOptions opts;
        opts.register_main_module = &reg;
        CHECK(load_image(good, arena, opts, li).ok());
        CHECK(win32::GetModuleHandleA("pgim_loader_test.exe") == li.base);
        CHECK(win32::GetModuleHandleA(nullptr) == li.base);
        const Baseline loaded = baseline_of(arena);
        LoadedImage second;
        opts.requested_base = 0x1'0000'0000ULL + 0x1000'0000;  // another base, same registration
        CHECK_CODE(load_image(good, arena, opts, second), ImageErrorCode::RegisterFailed);
        CHECK_BASELINE(arena, loaded);
        CHECK(second.base == 0);
        CHECK(unload_image(arena, li));
        CHECK_BASELINE(arena, base0);
    }
    // Registration is optional.
    CHECK(load(arena, good, li).ok());
    CHECK(unload_image(arena, li));
    CHECK_BASELINE(arena, base0);
}

struct ReadLog {
    std::vector<std::pair<uint64_t, size_t>> reads;
};

void test_streaming_reader() {
    // Big enough to need several payload chunks.
    Img img = fixture();
    img.secs[3].data = pattern(9, 0x400);
    const std::vector<uint8_t> bytes = write_container(img);

    ParsedImage parsed;
    CHECK(parse_image(bytes, parsed).ok());
    const uint64_t tables_end = 64 + 64 * 6 + 32;

    ReadLog log;
    MemoryImageReader mem(bytes);
    FunctionImageReader streaming(bytes.size(), [&](uint64_t off, void* dst, size_t len) {
        log.reads.emplace_back(off, len);
        return mem.read(off, dst, len);
    });
    VirtualMemoryArena a1;
    LoadedImage streamed;
    LoadOptions opts;
    opts.requested_base = 0x2'4000'0000ULL;
    CHECK(load_image(streaming, a1, opts, streamed).ok());

    VirtualMemoryArena a2;
    LoadedImage direct;
    CHECK(load_image(bytes, a2, opts, direct).ok());

    CHECK(streamed.base == direct.base && streamed.rebased && direct.rebased);
    CHECK(streamed.entry_point == direct.entry_point);
    CHECK(streamed.dir64_applied == direct.dir64_applied);
    CHECK(streamed.highlow_applied == direct.highlow_applied);
    CHECK(streamed.committed_bytes == direct.committed_bytes);
    CHECK(streamed.sections.size() == direct.sections.size());
    CHECK(streamed.page_runs.size() == direct.page_runs.size());
    VERIFY_MEMORY(a1, img, streamed);
    VERIFY_MEMORY(a2, img, direct);

    // Header and tables first, then one payload at a time in section order, each byte once.
    CHECK(log.reads.size() >= 4);
    CHECK(log.reads[0].first == 0 && log.reads[0].second == 64);
    size_t i = 1;
    while (i < log.reads.size() && log.reads[i].first + log.reads[i].second <= tables_end) ++i;
    CHECK(i >= 2);
    uint64_t payload_bytes = 0;
    uint64_t expected_bytes = 0;
    uint64_t last_end = 0;
    for (const SectionEntry& s : parsed.layout.sections) expected_bytes += s.payload_length;
    for (; i < log.reads.size(); ++i) {
        const auto [off, len] = log.reads[i];
        CHECK(off >= tables_end && off + len <= bytes.size());
        CHECK(off >= last_end);  // monotonic: no re-reads
        last_end = off + len;
        payload_bytes += len;
    }
    CHECK(payload_bytes == expected_bytes);
}

void test_hostile_reader() {
    const std::vector<uint8_t> good = write_container(fixture());
    // Header claims a 1 TiB container with 4G sections, but only the header can be served.
    std::vector<uint8_t> header(good.begin(), good.begin() + 64);
    patch64(header, 56, uint64_t{1} << 40);
    patch32(header, 44, 0xFFFFFFFF);
    uint64_t served = 0;
    FunctionImageReader hostile(uint64_t{1} << 40, [&](uint64_t off, void* dst, size_t len) {
        if (off + len > header.size()) return false;
        std::memcpy(dst, header.data() + off, len);
        served += len;
        return true;
    });
    ImageLayout layout;
    CHECK_CODE(parse_layout(hostile, layout), ImageErrorCode::ReadFailed);
    CHECK(layout.sections.empty() && served == 64);

    VirtualMemoryArena arena;
    LoadedImage li;
    LoadOptions opts;
    CHECK_CODE(load_image(hostile, arena, opts, li), ImageErrorCode::ReadFailed);
    CHECK(arena.reservation_count() == 0);

    // A reader that lies about its size cannot make the loader read out of bounds.
    MemoryImageReader mem(good);
    CHECK(!mem.read(good.size() - 1, header.data(), 2));
    CHECK(!mem.read(good.size() + 5, header.data(), 0 + 1));
    CHECK(mem.read(good.size(), header.data(), 0));
}

void test_large_sparse_image() {
    // 3.5 GiB guest image with two small islands and a >chunk-sized data section; the loader must
    // only commit what sections cover.
    Img img;
    img.base = 0x4'0000'0000ULL;
    img.size = 0xE000'0000u;
    img.secs = {
        {".data", 0x1000, 0xA1000, R | W, 0, pattern(1, 0xA0000), 0},
        {".far", 0xDFFF'0000u, 0x10000, R | W, 0, pattern(2, 0x20), 0},
    };
    for (uint32_t i = 0; i < 8192; ++i) img.dir64.push_back(0x1000 + 8 * i);
    for (uint32_t i = 0; i < 8192; ++i) img.dir64.push_back(0xDFFF'0000u + 8 * i);
    const std::vector<uint8_t> bytes = write_container(img);
    CHECK(bytes.size() < 0xC0000);  // tables + payloads only, not SizeOfImage

    VirtualMemoryArena arena;
    LoadedImage li;
    size_t payload_reads = 0;
    MemoryImageReader mem(bytes);
    FunctionImageReader streaming(bytes.size(), [&](uint64_t off, void* dst, size_t len) {
        if (off >= 64 + 64 * 2 + 4 * 16384) ++payload_reads;  // past the tables
        CHECK(len <= 256 * 1024);
        return mem.read(off, dst, len);
    });
    LoadOptions opts;
    opts.requested_base = 0x2'0000'0000ULL;  // negative delta
    CHECK(load_image(streaming, arena, opts, li).ok());
    CHECK(payload_reads >= 3);  // 640 KiB payload arrives in several range reads
    CHECK(li.rebased && li.dir64_applied == 16384 && li.highlow_applied == 0);
    CHECK(arena.reserved_bytes() == 0xE000'0000ULL);
    CHECK(arena.committed_bytes() == 0xA1000 + 0x10000);
    CHECK(li.committed_bytes == arena.committed_bytes());
    CHECK(li.page_runs.size() == 2);

    const uint64_t delta = li.base - img.base;
    std::vector<uint8_t> flat(0xA2000, 0);
    std::copy(img.secs[0].data.begin(), img.secs[0].data.end(), flat.begin() + 0x1000);
    bool ok = true;
    for (uint32_t i = 0; i < 8192; ++i) {
        const uint32_t rva = 0x1000 + 8 * i;
        uint64_t orig = 0;
        std::memcpy(&orig, flat.data() + rva, 8);  // little-endian host assumed by the test
        uint64_t now = 0;
        ok &= arena.read(li.base + rva, &now, 8) && now == orig + delta;
    }
    // .far: only the first 0x20 bytes are stored, the rest of its sites start as zero.
    for (uint32_t i = 0; i < 8192; ++i) {
        const uint32_t rva = 0xDFFF'0000u + 8 * i;
        uint8_t orig[8] = {};
        for (uint32_t k = 0; k < 8; ++k) {
            if (8 * i + k < img.secs[1].data.size()) orig[k] = img.secs[1].data[8 * i + k];
        }
        uint64_t o = 0;
        std::memcpy(&o, orig, 8);
        uint64_t now = 0;
        ok &= arena.read(li.base + rva, &now, 8) && now == o + delta;
    }
    check_impl(ok, "sparse image sites rebased", __LINE__);
    // The data is intact between the sites (page 0x80000 beyond the relocations: no sites there).
    std::vector<uint8_t> tail(0x1000);
    CHECK(arena.read(li.base + 0xA0000, tail.data(), tail.size()));
    CHECK(std::memcmp(tail.data(), img.secs[0].data.data() + 0x9F000, 0x1000) == 0);
    CHECK(arena.query(li.base + 0x10000000).state == MemState::Reserve);
    CHECK(unload_image(arena, li));
    CHECK(arena.reserved_bytes() == 0 && arena.committed_bytes() == 0);
}

// Container produced by the Rust writer (recomp/src/image.rs `write_image`): only the non-zero
// chunks are listed, everything between them is zero padding. Generated once by a throwaway
// `#[test]` in the Rust test module (not committed) that built this image and printed the bytes:
//   GuestImage { image_base: 0x1_4000_0000, size_of_image: 0x4000, section_alignment: 0x1000,
//     entry_point_rva: 0x1010, headers_mapped: true, executable_bytes: Keep, relocs_stripped: false,
//     sections: [<headers> rva 0 size 0x400 R data 0x40..0x4F; .text rva 0x1000 size 0x1800 RX
//       chars 0x60000020 data 1..=12; .data rva 0x3000 size 0x800 RW chars 0xC0000040 data
//       AA BB CC DD EE FF 11 22],
//     relocations: DIR64 0x1000, HIGHLOW 0x1008, HIGHLOW 0x3004 }
const uint8_t kGoldenTables[] = {
    0x50, 0x47, 0x49, 0x4D, 0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
    0x00, 0x10, 0x00, 0x00, 0x10, 0x10, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x08, 0x30, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x3C, 0x68, 0x65, 0x61, 0x64, 0x65, 0x72, 0x73,
    0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x04, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xF5, 0xE9, 0xCF, 0x8C, 0x5A, 0xA1, 0x8C, 0xA3,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2E, 0x74, 0x65, 0x78,
    0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x10, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x75, 0x20, 0x0E, 0x17,
    0x4F, 0x45, 0x2D, 0x22, 0x20, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00,
    0x2E, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x59, 0x93, 0x2B, 0xB8, 0xF8, 0x2C, 0x07, 0x9F, 0x40, 0x00, 0x00, 0xC0,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00,
    0x04, 0x30, 0x00, 0x00,
};

const uint8_t kGoldenPayload0[] = {
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B,
    0x4C, 0x4D, 0x4E, 0x4F,
};  // at offset 4096

const uint8_t kGoldenPayload1[] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
};  // at offset 8192

const uint8_t kGoldenPayload2[] = {
    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22,
};  // at offset 12288

void test_golden_rust_container() {
    constexpr size_t kGoldenSize = 12296;
    std::vector<uint8_t> golden(kGoldenSize, 0);
    std::memcpy(golden.data(), kGoldenTables, sizeof(kGoldenTables));
    std::memcpy(golden.data() + 4096, kGoldenPayload0, sizeof(kGoldenPayload0));
    std::memcpy(golden.data() + 8192, kGoldenPayload1, sizeof(kGoldenPayload1));
    std::memcpy(golden.data() + 12288, kGoldenPayload2, sizeof(kGoldenPayload2));

    ParsedImage parsed;
    CHECK(parse_image(golden, parsed).ok());
    const ImageHeader& h = parsed.layout.header;
    CHECK(h.image_base == 0x1'4000'0000ULL && h.size_of_image == 0x4000);
    CHECK(h.section_alignment == 0x1000 && h.entry_point_rva == 0x1010);
    CHECK(h.flags == kFlagHeadersMapped && h.section_count == 3);
    CHECK(h.dir64_count == 1 && h.highlow_count == 2 && h.file_size == kGoldenSize);
    CHECK(parsed.layout.dir64 == std::vector<uint32_t>({0x1000}));
    CHECK(parsed.layout.highlow == std::vector<uint32_t>({0x1008, 0x3004}));
    CHECK(parsed.layout.sections.size() == 3);
    if (parsed.layout.sections.size() == 3) {
        const SectionEntry& hd = parsed.layout.sections[0];
        const SectionEntry& tx = parsed.layout.sections[1];
        const SectionEntry& dt = parsed.layout.sections[2];
        CHECK(hd.name == "<headers>" && hd.rva == 0 && hd.virtual_size == 0x400 && hd.protection == R);
        CHECK(hd.is_headers() && hd.payload_offset == 4096 && hd.payload_length == 16);
        CHECK(tx.name == ".text" && tx.rva == 0x1000 && tx.virtual_size == 0x1800);
        CHECK(tx.protection == (R | X) && tx.characteristics == 0x60000020);
        CHECK(tx.payload_offset == 8192 && tx.payload_length == 12);
        CHECK(dt.name == ".data" && dt.rva == 0x3000 && dt.virtual_size == 0x800);
        CHECK(dt.protection == (R | W) && dt.characteristics == 0xC0000040);
        CHECK(dt.payload_offset == 12288 && dt.payload_length == 8);
        CHECK(!tx.is_headers() && !dt.bytes_omitted());
        CHECK(tx.checksum == fnv1a_64(kGoldenPayload1, sizeof(kGoldenPayload1)));
        CHECK(parsed.payloads[2].size() == 8 && parsed.payloads[2][0] == 0xAA);
    }

    // The C++ test writer produces the same bytes as the Rust writer.
    Img img;
    img.base = 0x1'4000'0000ULL;
    img.size = 0x4000;
    img.entry = 0x1010;
    img.flags = kFlagHeadersMapped;
    img.secs = {
        {"<headers>", 0, 0x400, R, kSectionFlagHeaders,
         std::vector<uint8_t>(kGoldenPayload0, kGoldenPayload0 + sizeof(kGoldenPayload0)), 0},
        {".text", 0x1000, 0x1800, R | X, 0,
         std::vector<uint8_t>(kGoldenPayload1, kGoldenPayload1 + sizeof(kGoldenPayload1)), 0x60000020},
        {".data", 0x3000, 0x800, R | W, 0,
         std::vector<uint8_t>(kGoldenPayload2, kGoldenPayload2 + sizeof(kGoldenPayload2)), 0xC0000040},
    };
    img.dir64 = {0x1000};
    img.highlow = {0x1008, 0x3004};
    CHECK(write_container(img) == golden);

    // Loading it rebases exactly as apply_relocations does.
    VirtualMemoryArena arena;
    LoadedImage li;
    CHECK(load(arena, golden, li, 0x1'4001'0000ULL).ok());
    CHECK(li.rebased && li.dir64_applied == 1 && li.highlow_applied == 2);
    uint64_t q = 0;
    uint32_t d = 0;
    CHECK(arena.read(li.base + 0x1000, &q, 8) && q == 0x0807060504030201ULL + 0x10000);
    CHECK(arena.read(li.base + 0x1008, &d, 4) && d == 0x0C0B0A09u + 0x10000);
    CHECK(arena.read(li.base + 0x3004, &d, 4) && d == 0x2211FFEEu + 0x10000);
    CHECK(li.entry_point == li.base + 0x1010);
    VERIFY_MEMORY(arena, img, li);
}

}  // namespace

int main() {
    test_primitives();
    test_parse_image_views();
    test_roundtrip_preferred_base();
    test_rebase();
    test_relocations_stripped();
    test_omitted_executable_bytes();
    test_protection_variants();
    test_corrupt_containers();
    test_cleanup_on_failure();
    test_streaming_reader();
    test_hostile_reader();
    test_large_sparse_image();
    test_golden_rust_container();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_image_loader failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_image_loader_tests passed.\n");
    return 0;
}
