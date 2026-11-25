// Host tests for the XCRDAPI.dll / GameInput.dll / xg_xs.dll guest interfaces (O23).
//
// Every interface call goes through a hand-written "lifted" caller (sub_vcall) that sets up the
// Win64 argument registers / stack slots and dispatches `call [rax + 8*slot]` through
// __playgta6_indirect_call, exactly like recompiled guest code. The zlib streams below were
// produced from the synthetic generators in this file (pattern_data / skewed_data).

#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#if defined(PLAYGTA6_HAVE_OPUS)
#include <opus.h>
#endif

#include "abi/playgta6_abi.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/gfx10_detile.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/xapu_opus.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xbox_scarlett.h"
#include "os/xbox_scarlett_guest.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xbox_scarlett_guest.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
namespace ga = playgta6::os::scarlett::guest_api;
namespace win32 = playgta6::os::win32;
using com::Guid;
using com::HResult;
using playgta6::os::VirtualMemoryArena;

// ---------------------------------------------------------------------------
// Synthetic data and zlib vectors
// ---------------------------------------------------------------------------

std::vector<uint8_t> pattern_data(size_t n) {
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>('a' + ((i * 31 + (i >> 3)) % 23));
    }
    return out;
}

std::vector<uint8_t> skewed_data(size_t n) {
    std::vector<uint8_t> out(n);
    uint64_t x = 12345;
    for (size_t i = 0; i < n; ++i) {
        x = (x * 1103515245u + 12345u) & 0x7fffffffu;
        const uint8_t b = static_cast<uint8_t>((x >> 16) & 0xFF);
        out[i] = (x & 7) == 0 ? b : static_cast<uint8_t>(b & 0x0F);
    }
    return out;
}

// zlib level 6 (dynamic Huffman) of pattern_data(4096)
const uint8_t kZlibDynamic[] = {
    0x78, 0x9c, 0xed, 0xce, 0x41, 0x02, 0x80, 0x10, 0x00, 0x45, 0xc1, 0xb3, 0xa2, 0x54, 0x84, 0x10,
    0x5d, 0xbf, 0x45, 0xef, 0x00, 0x1d, 0xe0, 0x6f, 0x67, 0x35, 0xe6, 0xa8, 0x36, 0x34, 0x17, 0xef,
    0x35, 0x0d, 0x9f, 0xe7, 0x76, 0x99, 0x0f, 0x4e, 0x60, 0x07, 0x16, 0xe0, 0x01, 0x3a, 0x50, 0x80,
    0x08, 0x6c, 0x80, 0x03, 0x26, 0xd0, 0x80, 0x0c, 0x04, 0xc0, 0x03, 0x16, 0x18, 0x40, 0x05, 0x12,
    0x40, 0xb3, 0xd3, 0x2c, 0x46, 0x6f, 0xbd, 0xf5, 0xd6, 0x5b, 0x6f, 0xbd, 0xf5, 0xd6, 0x5b, 0x6f,
    0xbd, 0xf5, 0xd6, 0x5b, 0x6f, 0xbd, 0xf5, 0xfe, 0xfd, 0x7e, 0x01, 0xea, 0xb8, 0xc0, 0x3b,
};
// zlib with fixed Huffman codes (Z_FIXED) of pattern_data(4096)
const uint8_t kZlibFixed[] = {
    0x78, 0x01, 0x4b, 0xcc, 0x2c, 0x4c, 0xca, 0x2a, 0x4a, 0xce, 0x2e, 0x49, 0xcd, 0x2d, 0x4d, 0xcb,
    0x2b, 0x4b, 0x2f, 0x48, 0x84, 0x08, 0xe4, 0x40, 0x05, 0x32, 0xa0, 0x02, 0x29, 0x50, 0x81, 0x72,
    0xa8, 0x40, 0x31, 0x54, 0x20, 0x1f, 0x2a, 0x90, 0x0d, 0x15, 0x48, 0x87, 0x0a, 0x24, 0x43, 0x05,
    0xca, 0xa0, 0x02, 0x45, 0x50, 0x81, 0x3c, 0xa8, 0x40, 0x16, 0x54, 0x20, 0x0d, 0x2a, 0x90, 0x04,
    0x15, 0x28, 0x85, 0x0a, 0x14, 0x42, 0x05, 0x72, 0xa1, 0x02, 0x50, 0x67, 0x16, 0x43, 0x9d, 0x99,
    0x9f, 0x38, 0xea, 0xee, 0x51, 0x77, 0x8f, 0xba, 0x7b, 0xd4, 0xdd, 0xa3, 0xee, 0x1e, 0x75, 0xf7,
    0xa8, 0xbb, 0x47, 0xdd, 0x3d, 0xea, 0xee, 0x51, 0x77, 0x8f, 0xba, 0x7b, 0xd4, 0xdd, 0xa3, 0xee,
    0x1e, 0x75, 0xf7, 0xa8, 0xbb, 0x89, 0x76, 0x37, 0x00, 0xea, 0xb8, 0xc0, 0x3b,
};
// zlib level 0 (stored block) of pattern_data(300)
const uint8_t kZlibStored[] = {
    0x78, 0x01, 0x01, 0x2c, 0x01, 0xd3, 0xfe, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x72, 0x63, 0x6b, 0x74,
    0x65, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x67, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x72, 0x63, 0x6c,
    0x74, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x68, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x72, 0x64,
    0x6c, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x77, 0x68, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x73,
    0x64, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6f, 0x77, 0x68, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6b,
    0x73, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x67, 0x6f, 0x77, 0x68, 0x70, 0x61, 0x69, 0x71, 0x63,
    0x6b, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x76, 0x67, 0x6f, 0x77, 0x68, 0x70, 0x61, 0x69, 0x72,
    0x63, 0x6b, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6e, 0x76, 0x67, 0x6f, 0x77, 0x68, 0x70, 0x61, 0x6a,
    0x72, 0x63, 0x6b, 0x73, 0x64, 0x6c, 0x74, 0x66, 0x6e, 0x76, 0x67, 0x6f, 0x77, 0x68, 0x70, 0x62,
    0x6a, 0x72, 0x63, 0x6b, 0x73, 0x64, 0x6c, 0x75, 0x66, 0x6e, 0x76, 0x67, 0x6f, 0x77, 0x68, 0x71,
    0x62, 0x6a, 0x72, 0x63, 0x6b, 0x73, 0x64, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x67, 0x6f, 0x77, 0x69,
    0x71, 0x62, 0x6a, 0x72, 0x63, 0x6b, 0x73, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x67, 0x6f, 0x61,
    0x69, 0x71, 0x62, 0x6a, 0x72, 0x63, 0x6b, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x67, 0x70,
    0x61, 0x69, 0x71, 0x62, 0x6a, 0x72, 0x63, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x76, 0x68,
    0x70, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x72, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6e, 0x77,
    0x68, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6a, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x66, 0x6f,
    0x77, 0x68, 0x70, 0x61, 0x69, 0x71, 0x62, 0x6b, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x75, 0x67,
    0x6f, 0x77, 0x68, 0x70, 0x61, 0x69, 0x71, 0x63, 0x6b, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6d, 0x76,
    0x67, 0x6f, 0x77, 0x68, 0x70, 0x61, 0x69, 0x72, 0x63, 0x6b, 0x73, 0x64, 0x6c, 0x74, 0x65, 0x6e,
    0x76, 0x67, 0x6f, 0x4b, 0x9c, 0x7e, 0x71,
};
// raw DEFLATE (no zlib header) of pattern_data(4096)
const uint8_t kRawDeflate[] = {
    0xed, 0xce, 0x41, 0x02, 0x80, 0x10, 0x00, 0x45, 0xc1, 0xb3, 0xa2, 0x54, 0x84, 0x10, 0x5d, 0xbf,
    0x45, 0xef, 0x00, 0x1d, 0xe0, 0x6f, 0x67, 0x35, 0xe6, 0xa8, 0x36, 0x34, 0x17, 0xef, 0x35, 0x0d,
    0x9f, 0xe7, 0x76, 0x99, 0x0f, 0x4e, 0x60, 0x07, 0x16, 0xe0, 0x01, 0x3a, 0x50, 0x80, 0x08, 0x6c,
    0x80, 0x03, 0x26, 0xd0, 0x80, 0x0c, 0x04, 0xc0, 0x03, 0x16, 0x18, 0x40, 0x05, 0x12, 0x40, 0xb3,
    0xd3, 0x2c, 0x46, 0x6f, 0xbd, 0xf5, 0xd6, 0x5b, 0x6f, 0xbd, 0xf5, 0xd6, 0x5b, 0x6f, 0xbd, 0xf5,
    0xd6, 0x5b, 0x6f, 0xbd, 0xf5, 0xfe, 0xfd, 0x7e, 0x01,
};
// zlib level 9 of skewed_data(2000): a skewed byte distribution, so some codes exceed 9 bits
const uint8_t kZlibSkewed[] = {
    0x78, 0xda, 0x15, 0x95, 0x5b, 0xe8, 0xa0, 0x65, 0x11, 0xc6, 0x67, 0xde, 0xc3, 0xcc, 0x3b, 0xf3,
    0xce, 0x01, 0x93, 0x6a, 0x6f, 0x16, 0xda, 0xba, 0x5a, 0x92, 0x82, 0xa0, 0x0d, 0xea, 0x2e, 0xb7,
    0x88, 0x0e, 0xa0, 0x04, 0x51, 0x50, 0x18, 0x05, 0x95, 0x0b, 0x41, 0xdd, 0x08, 0x42, 0x68, 0xd6,
    0x45, 0xec, 0x1a, 0x14, 0x41, 0x05, 0x4a, 0xa1, 0xbb, 0x14, 0x2e, 0xb4, 0x99, 0x54, 0xae, 0x6c,
    0xa5, 0xe0, 0x66, 0x50, 0x5a, 0xa0, 0xb0, 0x1d, 0x24, 0xd7, 0xbd, 0xa8, 0x28, 0x24, 0xe9, 0xbc,
    0x9d, 0xc8, 0xe6, 0x7f, 0xf9, 0xc1, 0xf7, 0x7d, 0xef, 0x3b, 0xcf, 0xf3, 0x7b, 0x9e, 0xb1, 0x31,
    0x35, 0xfd, 0xb0, 0xea, 0xb4, 0x1d, 0x79, 0x2d, 0x88, 0x39, 0xa4, 0xdc, 0x0f, 0x3e, 0x6c, 0x78,
    0x7b, 0x25, 0xad, 0xae, 0x2d, 0xf3, 0x9b, 0xbc, 0x04, 0x43, 0xe0, 0xf1, 0x35, 0xa7, 0x89, 0xe0,
    0x0b, 0x44, 0x2d, 0xd2, 0xe4, 0xb5, 0xbe, 0xc7, 0x52, 0xb1, 0xe3, 0xbd, 0x05, 0x65, 0xd7, 0x8f,
    0xd6, 0x47, 0xf5, 0x5e, 0x5e, 0x21, 0x36, 0x1b, 0xdc, 0x8e, 0xb9, 0x81, 0xe8, 0x90, 0x7f, 0x10,
    0xec, 0x4e, 0xa3, 0xbd, 0x68, 0xfa, 0xee, 0x3c, 0xf5, 0x84, 0x2e, 0x8d, 0xfa, 0xf4, 0xcd, 0xd1,
    0xa3, 0xeb, 0x82, 0x67, 0xd6, 0x5e, 0xa9, 0xa3, 0x3d, 0x65, 0x8b, 0xb7, 0xb1, 0x7e, 0x67, 0xb5,
    0xb9, 0xb7, 0xc3, 0x27, 0x78, 0x51, 0x36, 0x5f, 0xd7, 0xa0, 0x44, 0x9f, 0xbb, 0x3d, 0xc6, 0xa1,
    0x38, 0x68, 0xfd, 0x42, 0x6c, 0xec, 0x90, 0xf9, 0xe7, 0x36, 0x70, 0xa8, 0xe9, 0xd9, 0x35, 0x26,
    0x7b, 0x5b, 0x2d, 0xe7, 0x36, 0x51, 0x7b, 0x6e, 0x46, 0x2e, 0x05, 0xfb, 0x3e, 0x05, 0x6f, 0xdd,
    0xfe, 0x30, 0xf7, 0xde, 0x2c, 0xf3, 0xb4, 0x71, 0x9f, 0x6d, 0x3b, 0x4f, 0xc8, 0x25, 0x9a, 0x57,
    0xf5, 0x60, 0x8a, 0xa1, 0x1f, 0x99, 0xd1, 0x7d, 0x99, 0xff, 0x80, 0xb7, 0xd4, 0xb5, 0xfb, 0xf1,
    0x54, 0x1b, 0x08, 0xf4, 0xd0, 0x32, 0x4c, 0x8c, 0xf5, 0xb0, 0x84, 0xc2, 0x06, 0x3d, 0x41, 0x42,
    0x90, 0x93, 0x2f, 0xd4, 0x9c, 0x7d, 0xcb, 0x7a, 0x29, 0x01, 0xed, 0x85, 0xf0, 0x2b, 0x97, 0xd6,
    0xf6, 0x88, 0x47, 0x9b, 0x62, 0x1b, 0x01, 0x9f, 0x1c, 0xac, 0x09, 0xe6, 0xaf, 0x8a, 0x36, 0x65,
    0xd1, 0x7a, 0x92, 0xa5, 0x1b, 0x46, 0x7f, 0xa6, 0x8d, 0xd6, 0x4a, 0x98, 0xfb, 0xeb, 0xff, 0x41,
    0x3d, 0xae, 0xa5, 0xcd, 0xc0, 0x63, 0x3f, 0x8c, 0x73, 0xd0, 0x5c, 0xf1, 0x14, 0x60, 0x63, 0xcc,
    0x7e, 0x53, 0xb9, 0xb2, 0x4a, 0xe3, 0x67, 0x57, 0xeb, 0x92, 0x2d, 0xcf, 0xe3, 0x0c, 0x4e, 0x91,
    0x0b, 0x7d, 0x95, 0xdc, 0xcb, 0xef, 0x73, 0x70, 0x5a, 0xbb, 0x9d, 0xd5, 0x44, 0x23, 0xb5, 0x4b,
    0x80, 0xbc, 0x58, 0xf6, 0xb9, 0x8e, 0xde, 0x7d, 0xda, 0xbd, 0x38, 0x0c, 0x86, 0xcd, 0x53, 0x53,
    0x15, 0x67, 0xdf, 0xaf, 0xa6, 0xba, 0xd8, 0x04, 0xbf, 0xc8, 0x65, 0xd7, 0x84, 0xf5, 0x41, 0x66,
    0x88, 0x06, 0xfd, 0x2e, 0xdc, 0x94, 0x83, 0xfa, 0xe5, 0x9d, 0xae, 0x48, 0x74, 0xad, 0x68, 0x23,
    0x01, 0x9d, 0xd2, 0xa1, 0x26, 0x98, 0x7f, 0xa2, 0x6d, 0x25, 0x90, 0xfd, 0x9b, 0x4b, 0xce, 0x3e,
    0xf2, 0x41, 0xd0, 0x06, 0xcd, 0xe5, 0xde, 0xc8, 0x31, 0x06, 0xf6, 0xaf, 0x1b, 0x0e, 0xe9, 0x04,
    0x77, 0x37, 0xef, 0xda, 0x31, 0x2f, 0xf6, 0x0d, 0x49, 0x19, 0xef, 0xf0, 0x3e, 0x9d, 0x63, 0x3c,
    0xb9, 0x27, 0x8f, 0x36, 0xc6, 0x2f, 0x47, 0x2c, 0x94, 0x91, 0xef, 0xdc, 0x6b, 0x5a, 0x24, 0x1c,
    0x49, 0x56, 0xf5, 0x84, 0xcf, 0xee, 0xb4, 0x12, 0xb6, 0xfd, 0xe4, 0xc0, 0x36, 0x5b, 0x74, 0x1b,
    0x92, 0x24, 0x8e, 0x75, 0xac, 0xf0, 0x6b, 0xae, 0xf0, 0xc7, 0xc1, 0x1b, 0xb8, 0xe3, 0x49, 0xf7,
    0x92, 0xb1, 0xd9, 0x7b, 0x77, 0x74, 0x10, 0xf2, 0xaf, 0x39, 0x36, 0x0f, 0xd9, 0x1e, 0x54, 0xbc,
    0x4d, 0xbd, 0x6e, 0x6c, 0x22, 0xb7, 0x22, 0x48, 0x36, 0xb9, 0xf3, 0xa7, 0x17, 0xcd, 0xa9, 0x5d,
    0x06, 0x34, 0x2d, 0x63, 0xc7, 0x13, 0xe8, 0xe6, 0xdb, 0xd6, 0xe7, 0x22, 0x8a, 0x74, 0xcc, 0xe7,
    0x6a, 0x2e, 0x60, 0x1c, 0x7f, 0x18, 0x53, 0xd2, 0xfa, 0xba, 0x23, 0x76, 0xef, 0x64, 0xf6, 0x62,
    0x9e, 0x07, 0x0a, 0xf9, 0x5f, 0x73, 0x65, 0xe9, 0x8d, 0x97, 0xa0, 0x0e, 0x6d, 0xe0, 0x1f, 0x40,
    0x47, 0x58, 0x1d, 0xcf, 0x11, 0xe9, 0x06, 0x5c, 0xdf, 0xae, 0xb9, 0x70, 0x4d, 0xbd, 0x12, 0x11,
    0x5d, 0x78, 0x9f, 0x2c, 0xb9, 0x83, 0xd5, 0x3e, 0x56, 0xf9, 0x59, 0x75, 0xe0, 0x9b, 0x5a, 0x1d,
    0xb6, 0x8d, 0x8e, 0x8c, 0x3e, 0x8c, 0xa3, 0x50, 0xce, 0xd5, 0x8d, 0xf9, 0x16, 0xe0, 0x72, 0x83,
    0xe6, 0x23, 0xa5, 0x5b, 0x03, 0x8a, 0x07, 0x2c, 0xbd, 0xf4, 0x9a, 0x9d, 0x1c, 0x38, 0x87, 0xfe,
    0xdc, 0xc9, 0x7a, 0x0c, 0x79, 0x3d, 0xa6, 0x32, 0x6c, 0xb9, 0xcf, 0x04, 0x0a, 0x1c, 0xf8, 0xcc,
    0x44, 0x18, 0x8b, 0xed, 0x6c, 0x82, 0x20, 0xa5, 0xfc, 0x54, 0xbc, 0x03, 0x0e, 0x38, 0xec, 0x42,
    0x58, 0x40, 0x7d, 0xab, 0x9e, 0x20, 0xeb, 0x52, 0x61, 0x73, 0x0a, 0xe4, 0x17, 0xf6, 0x90, 0x59,
    0x39, 0x3a, 0xca, 0x3c, 0x11, 0x97, 0x7e, 0x71, 0xf5, 0x5d, 0x38, 0x94, 0x09, 0xfd, 0xc0, 0xf9,
    0xfe, 0x55, 0xd4, 0x11, 0x12, 0xf0, 0x9b, 0xbe, 0x57, 0xc5, 0x5f, 0x6e, 0xf0, 0x3e, 0x98, 0x23,
    0x2e, 0xb6, 0xbd, 0xa2, 0xe3, 0xb6, 0xd2, 0xb1, 0xd5, 0x59, 0x1f, 0xe2, 0x98, 0x73, 0x0e, 0xb9,
    0xba, 0x98, 0xfb, 0xca, 0xb8, 0x91, 0xa3, 0x52, 0x8e, 0xed, 0x2f, 0x73, 0x91, 0xae, 0xe9, 0x3f,
    0x32, 0x1e, 0x5a, 0xd2, 0x9d, 0x6e, 0x2c, 0xe5, 0xf7, 0xfc, 0x71, 0x41, 0xe2, 0x4e, 0x71, 0xf3,
    0xc8, 0x88, 0x9a, 0xf3, 0xb0, 0x46, 0x40, 0xf9, 0x7e, 0xa2, 0xc2, 0x01, 0x36, 0xed, 0x38, 0xd6,
    0xd4, 0x43, 0xf8, 0xe4, 0xe4, 0x4d, 0x20, 0xeb, 0x53, 0x6b, 0x2b, 0x14, 0x27, 0x1f, 0x6e, 0x06,
    0x2d, 0x4c, 0x9f, 0xa6, 0xae, 0x83, 0x61, 0xde, 0x8c, 0xd4, 0x25, 0x56, 0x26, 0xd4, 0x78, 0xae,
    0xfa, 0x80, 0x14, 0xcd, 0x2c, 0x7c, 0x26, 0x73, 0x67, 0x4c, 0x7c, 0x39, 0x12, 0x66, 0x2a, 0x3e,
    0x2d, 0xd5, 0x23, 0x2d, 0xf4, 0xa6, 0xc4, 0x45, 0xc5, 0xfd, 0x83, 0xb3, 0x70, 0xaa, 0xfc, 0x3c,
    0xb1, 0x43, 0x9b, 0x30, 0xff, 0x67, 0xb3, 0x6f, 0x18, 0xfb, 0x94, 0xd1, 0x64, 0x15, 0xb8, 0x0e,
    0x46, 0x05, 0x9f, 0xdb, 0x6d, 0x95, 0xc9, 0x45, 0xbc, 0xee, 0x74, 0xaf, 0xe6, 0xea, 0xf4, 0x56,
    0xe6, 0xb6, 0x86, 0xd9, 0xe1, 0x15, 0x63, 0x68, 0xa7, 0x1b, 0xa1, 0x2e, 0x62, 0xdd, 0x4e, 0x0b,
    0x16, 0xd7, 0xbd, 0x9d, 0x57, 0x5f, 0x38, 0x89, 0xef, 0x99, 0x2d, 0x09, 0x7d, 0xbd, 0x6f, 0x73,
    0x55, 0x0d, 0xda, 0xbb, 0xa7, 0x51, 0x81, 0xcb, 0xbf, 0xdf, 0x11, 0xd2, 0x53, 0x7f, 0x96, 0x6b,
    0x65, 0xdf, 0x79, 0x01, 0xd3, 0xb1, 0xd0, 0x7d, 0x6c, 0x8f, 0x0a, 0x12, 0xc5, 0xdf, 0xc7, 0x70,
    0x61, 0xb0, 0x47, 0x42, 0x4a, 0x30, 0x1e, 0xb7, 0xeb, 0xf2, 0x2a, 0x8e, 0x00, 0xec, 0x65, 0xb7,
    0xe6, 0x3f, 0x77, 0xfd, 0x46, 0x80, 0x4e, 0xa9, 0x4d, 0x30, 0x89, 0x1f, 0x86, 0x08, 0x09, 0xe2,
    0x11, 0xc8, 0x22, 0x3f, 0x91, 0xc5, 0xca, 0x61, 0x8a, 0xb7, 0x6c, 0x51, 0x69, 0x4c, 0x67, 0x86,
    0x6a, 0xd5, 0x6e, 0xbc, 0x11, 0x9b, 0x52, 0xba, 0xbd, 0x81, 0x72, 0x17, 0x1e, 0x79, 0x8d, 0x2c,
    0x5b, 0x73, 0xf7, 0xd7, 0x50, 0x15, 0x50, 0xef, 0x79, 0xab, 0x0e, 0x44, 0x33, 0xf9, 0xc6, 0xae,
    0x8b, 0x76, 0xcd, 0xff, 0xee, 0x2a, 0x9a, 0xa2, 0xf1, 0xd4, 0xf6, 0x55, 0xcd, 0x67, 0xcf, 0x8e,
    0x5d, 0x59, 0x1f, 0xf9, 0xf1, 0x58, 0x36, 0x9d, 0xe8, 0x15, 0xbb, 0xdc, 0x73, 0xf4, 0x8b, 0x9a,
    0x25, 0x27, 0xda, 0x9d, 0x8c, 0x51, 0xa3, 0xd1, 0x99, 0x95, 0xb4, 0x5d, 0xfd, 0xee, 0x6c, 0x9b,
    0xfa, 0xd4, 0xbf, 0x39, 0x0a, 0xba, 0xf2, 0x97, 0x76, 0xf5, 0x70, 0x47, 0x7e, 0x0f, 0xb6, 0x2c,
    0x6a, 0xf6, 0xed, 0xa8, 0x07, 0x3f, 0xf2, 0x63, 0x5a, 0x7d, 0x9e, 0x4e, 0x87, 0x2a, 0x87, 0x55,
    0x44, 0x7a, 0xae, 0x5e, 0x8f, 0xa4, 0xf5, 0x5b, 0xb6, 0x9d, 0xbb, 0xf9, 0x3d, 0x9b, 0x76, 0x6b,
    0x03, 0x8e, 0x96, 0x07, 0xd5, 0xdb, 0xf4, 0x5d, 0xb0, 0x4a, 0x10, 0xf8, 0xef, 0x36, 0xd8, 0x2a,
    0x4f, 0xde, 0x5e, 0xe5, 0x06, 0x30, 0xe3, 0x8a, 0xcf, 0xda, 0x1a, 0xd0, 0x4f, 0xce, 0x51, 0x0e,
    0x43, 0xbb, 0xab, 0xe7, 0xa8, 0x45, 0x40, 0x37, 0x8c, 0x5d, 0xad, 0xa8, 0xfc, 0xae, 0x21, 0xa4,
    0xcc, 0x7a, 0x7d, 0xed, 0x93, 0xe9, 0x19, 0x87, 0x6a, 0x8b, 0xec, 0xd9, 0x4a, 0x3f, 0xca, 0xac,
    0xc5, 0x73, 0xa6, 0x4e, 0x6f, 0x1d, 0xda, 0xbf, 0xa8, 0xea, 0xae, 0x14, 0xbd, 0xb5, 0x69, 0x6d,
    0x1f, 0x1b, 0x8f, 0x1e, 0xb4, 0xe1, 0x08, 0xbb, 0xcc, 0xd3, 0xab, 0x23, 0xc6, 0x45, 0x18, 0x2d,
    0x9a, 0xd9, 0x25, 0xe1, 0x56, 0x18, 0xc0, 0xfb, 0x6b, 0xc9, 0xb8, 0x1a, 0xbf, 0x0c, 0x6d, 0xd4,
    0xca, 0xe3, 0xef, 0xd5, 0xf2, 0xc0, 0x26, 0x71, 0xfd, 0x8c, 0x2a, 0xf0, 0x25, 0xe7, 0xf7, 0xb2,
    0xc0, 0x88, 0x3b, 0x5c, 0x62, 0x16, 0x23, 0x57, 0x1b, 0xda, 0xcc, 0xb9, 0x7e, 0x1d, 0xb5, 0x88,
    0xba, 0xf2, 0x65, 0xe8, 0x05, 0xe0, 0x96, 0x43, 0x38, 0xdb, 0xa0, 0xc4, 0xcf, 0xeb, 0x41, 0x6f,
    0x0a, 0x3e, 0x64, 0x5b, 0x49, 0xa4, 0x3d, 0xbf, 0x72, 0x2a, 0xed, 0xfc, 0x1f, 0x93, 0x78, 0xad,
    0x88, 0xaf, 0xc0, 0x41, 0xbf, 0x1a, 0x3e, 0x5f, 0xe5, 0x6f, 0xea, 0xe3, 0x05, 0xed, 0xab, 0x4a,
    0x61, 0xbd, 0x6d, 0x1e, 0x8c, 0xe7, 0xed, 0x75, 0x5e, 0x78, 0x57, 0x71, 0x7d, 0x99, 0x51, 0x21,
    0xb3, 0x1d, 0x85, 0x0a, 0x49, 0x9b, 0xf6, 0x92, 0x61, 0x55, 0x2e, 0x31, 0x6e, 0x51, 0x9f, 0x09,
    0x21, 0x8f, 0x8f, 0xff, 0x03, 0x3c, 0xe1, 0xbb, 0xb3,
};
// zlib level 1 of pattern_data(65536): one 64 KiB page
const uint8_t kZlibPage64k[] = {
    0x78, 0x01, 0xed, 0xdd, 0x5b, 0x52, 0xc3, 0x40, 0x10, 0x43, 0xd1, 0xb5, 0x26, 0x40, 0x42, 0xc2,
    0xc3, 0x21, 0x2f, 0xb6, 0x8f, 0x65, 0xf5, 0x22, 0xf8, 0x38, 0xbf, 0x53, 0x54, 0xc5, 0x45, 0xd9,
    0xd3, 0xa3, 0x2b, 0x75, 0xcf, 0xee, 0xf4, 0xb3, 0x3f, 0x5f, 0x5f, 0x3e, 0xee, 0x6f, 0x5f, 0x8f,
    0xc3, 0xf7, 0xf3, 0x78, 0xd9, 0x75, 0xe1, 0x73, 0x16, 0xde, 0x67, 0xe1, 0x75, 0x16, 0x7e, 0x67,
    0xe1, 0x36, 0x0b, 0xcb, 0x2c, 0x7c, 0xcc, 0xc2, 0x71, 0x16, 0x5e, 0x66, 0xe1, 0x39, 0x0b, 0xeb,
    0x4f, 0x6c, 0x7f, 0xb1, 0xfe, 0xc4, 0xf6, 0x17, 0xf9, 0xcd, 0x2c, 0xe4, 0x37, 0xb3, 0xb0, 0x3d,
    0xc4, 0xba, 0xb0, 0x3d, 0xc4, 0xba, 0xd0, 0xa7, 0xba, 0xbd, 0xf6, 0xa9, 0x96, 0xdf, 0x79, 0xcc,
    0xdb, 0x3c, 0xe6, 0x32, 0x8f, 0xe9, 0xb9, 0xfd, 0xbf, 0xb7, 0x37, 0xc9, 0x7b, 0xd2, 0x0f, 0x75,
    0x3e, 0xc3, 0xc3, 0xbc, 0x16, 0x7b, 0xdf, 0x65, 0x37, 0x18, 0xfb, 0x49, 0x37, 0x74, 0xfb, 0xb7,
    0xba, 0xa3, 0xce, 0x9f, 0x7e, 0x9c, 0x4f, 0x7a, 0xf2, 0x72, 0xae, 0xea, 0xc1, 0x56, 0xbd, 0xec,
    0x81, 0xdc, 0xf9, 0xbb, 0xca, 0x82, 0xde, 0xa1, 0xd3, 0xe8, 0xe2, 0xaf, 0xc7, 0xc8, 0x77, 0xf5,
    0x72, 0x36, 0x46, 0xf5, 0x52, 0xbd, 0xcc, 0x59, 0x01, 0xaf, 0x2a, 0x89, 0xc3, 0x07, 0x71, 0xcd,
    0x10, 0x16, 0x3c, 0xb6, 0x27, 0x46, 0xf5, 0xb2, 0x64, 0x1f, 0x8f, 0xc5, 0x63, 0xf9, 0x3b, 0xfc,
    0x34, 0x9c, 0x0d, 0x67, 0x8b, 0x66, 0xe2, 0x4b, 0xf1, 0xb9, 0xa3, 0x99, 0xe4, 0x0a, 0x4a, 0x9a,
    0xe5, 0x21, 0x0a, 0x54, 0xe8, 0x06, 0xba, 0x21, 0x9a, 0x89, 0x3f, 0xcf, 0x9f, 0x4f, 0x36, 0x41,
    0x2e, 0x4c, 0x0e, 0x2f, 0xca, 0x99, 0x6e, 0xa0, 0x1b, 0xe8, 0x06, 0xf9, 0xd8, 0x89, 0x1f, 0xd3,
    0x0d, 0x03, 0x96, 0xe9, 0x06, 0xba, 0x21, 0x9a, 0x49, 0xae, 0x57, 0xae, 0x37, 0x9a, 0x49, 0x4e,
    0xa9, 0x1b, 0x23, 0xdd, 0x40, 0x37, 0xd0, 0x0d, 0xfa, 0xa5, 0xf4, 0x37, 0xe8, 0x6f, 0xd0, 0xdf,
    0x80, 0x9f, 0xc8, 0x9f, 0xc8, 0x9f, 0xc8, 0x9f, 0xc8, 0x9f, 0xe0, 0xc8, 0x38, 0x32, 0x8e, 0x8c,
    0x23, 0xe3, 0xc8, 0x4d, 0x60, 0xcd, 0xf8, 0x8d, 0x13, 0x8e, 0x8c, 0x23, 0xe3, 0xc8, 0xe6, 0xcd,
    0xe8, 0xdf, 0xd1, 0xbf, 0xa3, 0x7f, 0x87, 0x9f, 0x36, 0xed, 0xad, 0x72, 0x78, 0xe6, 0x9c, 0x6d,
    0x01, 0x65, 0x7e, 0x1a, 0x3f, 0x8d, 0x9f, 0xc6, 0x4f, 0xe3, 0xa7, 0xf1, 0xd3, 0xf8, 0x69, 0xfc,
    0x34, 0x7e, 0x1a, 0x3f, 0x8d, 0x9f, 0xc6, 0x4f, 0xe3, 0xa7, 0xf1, 0xd3, 0xf8, 0x69, 0xfc, 0x34,
    0x7e, 0x1a, 0x3f, 0x2d, 0x5e, 0x62, 0xf6, 0x82, 0x78, 0x89, 0xe6, 0xf2, 0xd7, 0x49, 0xd1, 0x97,
    0xa1, 0x2f, 0x43, 0x5f, 0xc6, 0xfd, 0xad, 0x6f, 0x81, 0x7e, 0x6e, 0xf7, 0xaa, 0x94, 0x20, 0xd2,
    0x0d, 0x74, 0x03, 0xdd, 0x40, 0x37, 0xd0, 0x0d, 0x74, 0x03, 0xdd, 0x60, 0x1e, 0x5e, 0xab, 0xa1,
    0x39, 0xda, 0xe6, 0xe1, 0x99, 0x87, 0x87, 0x9f, 0x3c, 0xdc, 0x57, 0x37, 0x81, 0x2b, 0x39, 0xbc,
    0x4e, 0x84, 0x34, 0xd7, 0xc2, 0x5c, 0x8b, 0x7c, 0x12, 0xf8, 0x09, 0x7e, 0x12, 0x76, 0x24, 0x7f,
    0x22, 0x7f, 0x22, 0x7f, 0x22, 0x7f, 0x82, 0x23, 0xe3, 0xc8, 0x38, 0x32, 0x8e, 0x8c, 0x23, 0xe3,
    0xc8, 0x38, 0x32, 0x8e, 0x8c, 0x23, 0x37, 0x81, 0x85, 0x23, 0xe3, 0xc8, 0x38, 0x32, 0x8e, 0xec,
    0x5e, 0x5a, 0xf7, 0xd2, 0x26, 0x89, 0x9a, 0xbd, 0x20, 0x0c, 0x5d, 0x0e, 0xaf, 0x01, 0x65, 0x1c,
    0x19, 0x47, 0xc6, 0x91, 0x0f, 0xdf, 0xfa, 0xb9, 0xbb, 0x31, 0xd2, 0x0d, 0x74, 0x43, 0x34, 0x93,
    0x79, 0x78, 0xfa, 0x77, 0x72, 0x62, 0xac, 0x7a, 0x34, 0x07, 0x4a, 0xfe, 0x44, 0xfe, 0x84, 0x6e,
    0x38, 0x2e, 0x7d, 0x0b, 0xe4, 0x4f, 0xe6, 0xda, 0x31, 0xf7, 0x31, 0xce, 0xc6, 0xd8, 0x53, 0x93,
    0x7e, 0x57, 0xf5, 0xb2, 0x84, 0x45, 0xbd, 0x54, 0x2f, 0xd5, 0x4b, 0xf5, 0x52, 0xdf, 0xbf, 0xbe,
    0x7f, 0x7d, 0xff, 0xfc, 0x06, 0xba, 0xa1, 0x9d, 0x1b, 0x74, 0x43, 0x1b, 0x7a, 0xe4, 0xd6, 0xe5,
    0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5,
    0xd6, 0xe5, 0xd6, 0xe5, 0xd6, 0xe5, 0x4f, 0xe4, 0x4f, 0xe4, 0x4f, 0xcc, 0x8f, 0x6d, 0x35, 0x94,
    0x3f, 0x91, 0x5b, 0x97, 0x5b, 0x97, 0x5b, 0x97, 0x2b, 0x90, 0x2b, 0x90, 0x2b, 0x48, 0xa6, 0xc2,
    0xbd, 0xb4, 0xfa, 0x77, 0xe2, 0x28, 0xf3, 0xd3, 0xf8, 0x69, 0xf1, 0x12, 0xe7, 0x9a, 0x09, 0x73,
    0xa0, 0x66, 0x63, 0xa4, 0x1b, 0xda, 0xb9, 0x41, 0x37, 0xd0, 0x0d, 0x74, 0x03, 0xdd, 0x40, 0x37,
    0xd0, 0x0d, 0x74, 0x83, 0xbe, 0xff, 0x56, 0x43, 0xba, 0x41, 0x1e, 0x59, 0x1e, 0x59, 0x1e, 0x59,
    0x1e, 0x19, 0x3f, 0xc1, 0x4f, 0xf0, 0x93, 0xf3, 0xb5, 0x6f, 0x81, 0xf9, 0x27, 0x73, 0xcd, 0xc4,
    0x5d, 0xfe, 0x44, 0xfe, 0x44, 0xfe, 0x44, 0xfe, 0x04, 0x47, 0xc6, 0x91, 0xc3, 0xd0, 0xb3, 0x17,
    0xe0, 0x27, 0xf8, 0x09, 0x7e, 0x52, 0x83, 0x0d, 0x3f, 0xc1, 0x4f, 0xf0, 0x13, 0xfc, 0x04, 0x3f,
    0xc1, 0x4f, 0xf0, 0x13, 0xfc, 0xa4, 0xd5, 0xf0, 0xb2, 0xc3, 0x4f, 0x3a, 0xd1, 0x40, 0x1f, 0xa3,
    0x3e, 0x46, 0x7d, 0x8c, 0xfa, 0x18, 0x71, 0x64, 0x1c, 0x19, 0x47, 0xc6, 0x91, 0x71, 0x64, 0x1c,
    0x19, 0x47, 0x76, 0x9f, 0x40, 0xab, 0xa1, 0x3c, 0xb2, 0x3c, 0xb2, 0x3c, 0x32, 0x3f, 0x8d, 0x9f,
    0xc6, 0x4f, 0xe3, 0xa7, 0xc5, 0x4b, 0xcc, 0x5e, 0x80, 0x23, 0xe3, 0xc8, 0x38, 0x72, 0x3b, 0x79,
    0xdd, 0xc7, 0xf8, 0x3f, 0xee, 0x63, 0xfc, 0x03, 0x31, 0xd9, 0x06, 0x45,
};

// ---------------------------------------------------------------------------
// Guest-convention caller
// ---------------------------------------------------------------------------

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

uint32_t load32(uint64_t va) {
    uint32_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

void store64(uint64_t va, uint64_t value) {
    CHECK(g_arena->write(va, &value, sizeof(value)));
}

uint64_t& gpr(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

constexpr uint64_t kVCall = 0x140010000ULL;
constexpr uint64_t kGuestDeviceCallback = 0x140010100ULL;

struct Arg {
    uint64_t bits;
    bool is_float;
};

struct VCallSpec {
    uint64_t target = 0;  // 0: virtual call through [args[0]] + 8*slot
    uint32_t slot = 0;
    std::vector<Arg> args;
};

thread_local VCallSpec t_spec;

void sub_vcall(GuestState* s) {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    const size_t n = t_spec.args.size();
    const size_t stack_n = n > 4 ? n - 4 : 0;
    const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
    gpr(s, GuestReg::Rsp) -= frame;
    const uint64_t rsp = gpr(s, GuestReg::Rsp);
    for (size_t i = 0; i < n; ++i) {
        const Arg& a = t_spec.args[i];
        if (i < 4) {
            if (a.is_float) {
                s->xmm[i][0] = a.bits;
            } else {
                gpr(s, kRegs[i]) = a.bits;
            }
        } else {
            store64(rsp + 0x20 + 8 * (i - 4), a.bits);
        }
    }
    uint64_t target = t_spec.target;
    if (target == 0) {
        target = load64(load64(gpr(s, GuestReg::Rcx)) + 8ull * t_spec.slot);
    }
    gpr(s, GuestReg::Rsp) -= 8;
    store64(gpr(s, GuestReg::Rsp), kVCall + 0x40);
    __playgta6_indirect_call(s, target);
    CHECK(gpr(s, GuestReg::Rsp) == rsp);
    gpr(s, GuestReg::Rsp) += frame + 8;
}

// GameInput device callback (token, context, device, timestamp, current, previous): records the
// arguments; the stack arguments are read from the caller's frame.
struct DeviceCallbackRecord {
    std::atomic<int> calls{0};
    uint64_t token = 0;
    uint64_t context = 0;
    uint64_t device = 0;
    uint64_t timestamp = 0;
    uint32_t current = 0;
    uint32_t previous = 0;
};
DeviceCallbackRecord g_device_cb;

void sub_guest_device_callback(GuestState* s) {
    const uint64_t rsp = gpr(s, GuestReg::Rsp);
    g_device_cb.token = gpr(s, GuestReg::Rcx);
    g_device_cb.context = gpr(s, GuestReg::Rdx);
    g_device_cb.device = gpr(s, GuestReg::R8);
    g_device_cb.timestamp = gpr(s, GuestReg::R9);
    g_device_cb.current = static_cast<uint32_t>(load64(rsp + kGuestStackArgOffset));
    g_device_cb.previous = static_cast<uint32_t>(load64(rsp + kGuestStackArgOffset + 8));
    g_device_cb.calls.fetch_add(1);
    gpr(s, GuestReg::Rsp) += 8;
}

const LiftedFunctionEntry kFunctions[] = {
    {kVCall, sub_vcall},
    {kGuestDeviceCallback, sub_guest_device_callback},
};

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

bool diag_contains(const char* needle) {
    for (const std::string& d : com::diagnostics()) {
        if (d.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

Arg I(uint64_t v) {
    return Arg{v, false};
}

Arg F(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return Arg{bits, true};
}

int32_t as_hr(uint64_t rax) {
    return static_cast<int32_t>(static_cast<uint32_t>(rax));
}

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        com::set_guest_memory(&arena);
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 512 * 1024, thread));
        os::scarlett::reset_dstorage_for_testing();
        ga::register_xbox_scarlett_guest_exports();
    }

    ~Fixture() {
        ga::reset_xbox_scarlett_guest();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        os::scarlett::reset_dstorage_for_testing();
        g_arena = nullptr;
    }

    GuestState& state() { return thread.state; }

    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<Arg> args = {}) {
        return vcall_on(thread, obj, slot, args);
    }

    uint64_t vcall_on(GuestThread& t, uint64_t obj, uint32_t slot, std::initializer_list<Arg> args) {
        t_spec = VCallSpec{};
        t_spec.slot = slot;
        t_spec.args.push_back({obj, false});
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, t.state, kVCall);
    }

    uint64_t dcall(uint64_t target, std::initializer_list<Arg> args = {}) {
        t_spec = VCallSpec{};
        t_spec.target = target;
        t_spec.args.assign(args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t guid(const Guid& g) {
        const uint64_t va = com::guest_alloc(16);
        CHECK(com::write_guest(va, g));
        return va;
    }

    uint64_t wstr(const char16_t* s) {
        const size_t n = std::char_traits<char16_t>::length(s) + 1;
        const uint64_t va = com::guest_alloc(n * 2);
        CHECK(com::write_guest(va, s, n * 2));
        return va;
    }

    uint64_t cstr(const char* s) {
        const size_t n = std::strlen(s) + 1;
        const uint64_t va = com::guest_alloc(n);
        CHECK(com::write_guest(va, s, n));
        return va;
    }

    uint64_t out_slot() {
        const uint64_t va = com::guest_alloc(8);
        store64(va, 0xDEADBEEFDEADBEEFull);
        return va;
    }

    // Guest-callable address of an export registered by register_xbox_scarlett_guest_exports().
    uint64_t export_address(const char* module, const char* name) {
        const uint64_t proc = pg6_loader_resolve_import(module, name);
        CHECK(proc != 0);
        const uint64_t callable = guest_callable_for_proc(proc);
        CHECK(callable != 0);
        return callable;
    }

    uint64_t factory() {
        const uint64_t out = out_slot();
        const uint64_t get = export_address("XCRDAPI.dll", "DStorageGetFactory");
        CHECK(as_hr(dcall(get, {I(guid(ga::kIidDStorageFactory)), I(out)})) == 0);
        return load64(out);
    }

    uint64_t open_file(uint64_t factory, const char16_t* path, HResult* hr_out = nullptr) {
        const uint64_t out = out_slot();
        const HResult hr = as_hr(vcall(factory, 4, {I(wstr(path)), I(guid(ga::kIidDStorageFile)), I(out)}));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        return load64(out);
    }

    uint64_t create_queue(uint64_t factory, uint16_t capacity, uint64_t source_type = 0) {
        ga::DStorageQueueDesc desc{};
        desc.source_type = source_type;
        desc.capacity = capacity;
        desc.priority = 0;
        desc.name = cstr("test queue");
        desc.device = 1;
        const uint64_t desc_va = com::guest_alloc(sizeof(desc));
        CHECK(com::write_guest(desc_va, desc));
        const uint64_t out = out_slot();
        CHECK(as_hr(vcall(factory, 10, {I(desc_va), I(guid(ga::kIidDStorageQueue)), I(out)})) == 0);
        return load64(out);
    }

    uint64_t create_status(uint64_t factory, uint32_t capacity) {
        const uint64_t out = out_slot();
        CHECK(as_hr(vcall(factory, 5, {I(capacity), I(cstr("status")), I(guid(ga::kIidDStorageStatusArray)), I(out)})) ==
              0);
        return load64(out);
    }

    // Copies `r` to guest memory and calls EnqueueRequest (slot 3).
    void enqueue(uint64_t queue, const ga::DStorageRequest& r) {
        const uint64_t va = com::guest_alloc(sizeof(r));
        CHECK(com::write_guest(va, r));
        vcall(queue, 3, {I(va)});
        com::guest_free(va);
    }

    bool wait_complete(uint64_t status, uint32_t index, int timeout_ms = 10000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if ((vcall(status, 3, {I(index)}) & 0xFF) != 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }
};

ga::DStorageRequest file_request(uint64_t file, uint64_t offset, uint32_t size, uint64_t dest, uint32_t dest_size,
                                 uint64_t tag = 0) {
    ga::DStorageRequest r{};
    r.options = 0;
    r.destination = dest;
    r.destination_size = dest_size;
    r.source = file;
    r.source_offset = offset;
    r.source_size = size;
    r.cancellation_tag = tag;
    return r;
}

std::vector<uint8_t> guest_bytes(uint64_t va, size_t n) {
    std::vector<uint8_t> out(n);
    CHECK(com::read_guest(va, out.data(), n));
    return out;
}

// ---------------------------------------------------------------------------
// Inflate
// ---------------------------------------------------------------------------

template <size_t N>
bool inflates_to(const uint8_t (&stream)[N], const std::vector<uint8_t>& expected) {
    std::vector<uint8_t> out(expected.size() + 16);
    size_t n = 0;
    const ga::InflateStatus st = ga::inflate_stream(stream, N, out.data(), out.size(), &n);
    out.resize(n);
    return st == ga::InflateStatus::Ok && out == expected;
}

void test_inflate() {
    CHECK(inflates_to(kZlibDynamic, pattern_data(4096)));
    CHECK(inflates_to(kZlibFixed, pattern_data(4096)));
    CHECK(inflates_to(kZlibStored, pattern_data(300)));
    CHECK(inflates_to(kRawDeflate, pattern_data(4096)));
    CHECK(inflates_to(kZlibSkewed, skewed_data(2000)));
    CHECK(inflates_to(kZlibPage64k, pattern_data(65536)));

    std::vector<uint8_t> out(8192);
    size_t n = 0;
    // Corrupt Adler-32.
    std::vector<uint8_t> bad(std::begin(kZlibDynamic), std::end(kZlibDynamic));
    bad.back() ^= 0x01;
    CHECK(ga::inflate_stream(bad.data(), bad.size(), out.data(), out.size(), &n) == ga::InflateStatus::ChecksumMismatch);
    // Truncated stream.
    CHECK(ga::inflate_stream(kZlibDynamic, sizeof(kZlibDynamic) - 10, out.data(), out.size(), &n) ==
          ga::InflateStatus::Truncated);
    // Destination too small.
    CHECK(ga::inflate_stream(kZlibDynamic, sizeof(kZlibDynamic), out.data(), 4095, &n) ==
          ga::InflateStatus::OutputTooSmall);
    // Preset dictionary flag.
    const uint8_t dict[] = {0x78, 0xBB, 0, 0, 0, 0};
    CHECK(ga::inflate_stream(dict, sizeof(dict), out.data(), out.size(), &n) == ga::InflateStatus::BadHeader);
    // Reserved block type 3.
    const uint8_t reserved[] = {0x07, 0x00};
    CHECK(ga::inflate_stream(reserved, sizeof(reserved), out.data(), out.size(), &n) == ga::InflateStatus::BadData);
    // A back-reference before the start of the output.
    const uint8_t far_ref[] = {0x03, 0x02, 0x00};  // fixed block: length 3 distance 1 with no output yet
    CHECK(ga::inflate_stream(far_ref, sizeof(far_ref), out.data(), out.size(), &n) != ga::InflateStatus::Ok);
}

// ---------------------------------------------------------------------------
// DirectStorage
// ---------------------------------------------------------------------------

void test_factory_identity_and_stubs() {
    Fixture fx;
    const uint64_t get = fx.export_address("XCRDAPI.dll", "DStorageGetFactory");
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(get, {I(fx.guid(com::make_guid("00000000-1111-2222-3333-444444444444"))), I(out)})) ==
          com::E_NOINTERFACE_HR);
    CHECK(load64(out) == 0);
    CHECK(as_hr(fx.dcall(get, {I(fx.guid(ga::kIidDStorageFactory)), I(0)})) == com::E_POINTER_HR);

    const uint64_t f1 = fx.factory();
    const uint64_t f2 = fx.factory();
    CHECK(f1 != 0 && f1 == f2);  // one factory, a reference per call
    CHECK(fx.vcall(f1, 2) == 2);  // Release: 3 -> 2 (the module keeps one)
    CHECK(fx.vcall(f1, 2) == 1);

    // QueryInterface(IID_IUnknown) answers the identity.
    const uint64_t qi = fx.out_slot();
    CHECK(as_hr(fx.vcall(f1, 0, {I(fx.guid(com::IID_IUnknown_GUID)), I(qi)})) == 0);
    CHECK(load64(qi) == f1);
    fx.vcall(f1, 2);

    // Slot 3 is never called by the title: an unnamed stub that logs and traps.
    GuestFault fault{};
    bool faulted = false;
    try {
        fx.vcall(f1, 3, {I(1), I(2), I(3)});
    } catch (const GuestFault& f) {
        faulted = true;
        fault = f;
    }
    CHECK(faulted);
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot));
    CHECK(fault.info.detail == 3);
    CHECK(diag_contains("XCRDAPI.IDStorageFactory {62959D11-D45F-4884-A0A3-F7862EA243AE} slot 3"));
}

void test_open_file_and_information() {
    Fixture fx;
    std::vector<uint8_t> data = pattern_data(70000);
    os::scarlett::register_dstorage_virtual_file("G:\\data\\pg6\\stream.ttpf", data);
    const uint64_t factory = fx.factory();

    HResult hr = 0;
    CHECK(fx.open_file(factory, u"G:\\data\\pg6\\missing.ttpf", &hr) == 0);
    CHECK(hr == ga::kHrFileNotFound);
    // Wrong IID
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.vcall(factory, 4, {I(fx.wstr(u"G:\\data\\pg6\\stream.ttpf")), I(fx.guid(ga::kIidDStorageQueue)),
                                      I(out)})) == com::E_NOINTERFACE_HR);
    CHECK(load64(out) == 0);

    const uint64_t file = fx.open_file(factory, u"G:\\data\\pg6\\stream.ttpf", &hr);
    CHECK(hr == 0 && file != 0);
    const uint64_t info_va = com::guest_alloc(sizeof(ga::ByHandleFileInformation));
    CHECK(as_hr(fx.vcall(file, 4, {I(info_va)})) == 0);
    ga::ByHandleFileInformation info{};
    CHECK(com::read_guest(info_va, info));
    CHECK(info.file_size_high == 0 && info.file_size_low == 70000);
    CHECK(info.number_of_links == 1);
    fx.vcall(file, 3);  // Close
    CHECK(fx.vcall(file, 2) == 0);
    CHECK(fx.vcall(factory, 2) == 1);
}

void test_requests_round_trip() {
    Fixture fx;
    const std::vector<uint8_t> data = pattern_data(300000);
    os::scarlett::register_dstorage_virtual_file("G:\\data\\a.bin", data);
    const uint64_t factory = fx.factory();
    // SetStagingBufferSize (slot 8), as the title does right after creation.
    CHECK(as_hr(fx.vcall(factory, 8, {I(0x2000000)})) == 0);
    const uint64_t file = fx.open_file(factory, u"G:\\data\\a.bin");
    const uint64_t queue = fx.create_queue(factory, 127);
    const uint64_t status = fx.create_status(factory, 1);
    CHECK(file != 0 && queue != 0 && status != 0);
    CHECK((fx.vcall(status, 3, {I(0)}) & 0xFF) == 0);  // nothing enqueued yet: not complete
    CHECK(as_hr(fx.vcall(status, 4, {I(0)})) == ga::kHrPending);

    // 40 requests: runs of adjacent 4 KiB pages (coalesced), a few scattered ones, mixed order.
    const uint64_t dest = com::guest_alloc(512 * 1024);
    struct Expect {
        uint64_t offset;
        uint32_t size;
        uint64_t dest;
    };
    std::vector<Expect> expect;
    for (int i = 0; i < 40; ++i) {
        const uint64_t offset = (i % 4 == 3) ? 200000 + 7 * i : static_cast<uint64_t>((39 - i) * 4096);
        const uint32_t size = (i % 4 == 3) ? 777 : 4096;
        const uint64_t d = dest + static_cast<uint64_t>(i) * 8192;
        expect.push_back({offset, size, d});
        fx.enqueue(queue, file_request(file, offset, size, d, 8192));
    }
    fx.vcall(queue, 4, {I(status), I(0)});  // EnqueueStatus
    CHECK((fx.vcall(status, 3, {I(0)}) & 0xFF) == 0);
    fx.vcall(queue, 6);  // Submit
    CHECK(fx.wait_complete(status, 0));
    CHECK(as_hr(fx.vcall(status, 4, {I(0)})) == 0);
    for (const Expect& e : expect) {
        const std::vector<uint8_t> got = guest_bytes(e.dest, e.size);
        CHECK(std::memcmp(got.data(), data.data() + e.offset, e.size) == 0);
    }
    const ga::DStorageStats st = ga::dstorage_stats();
    CHECK(st.requests_completed == 40);
    CHECK(st.requests_failed == 0);
    CHECK(st.coalesced_requests >= 30);
    CHECK(st.file_reads < 20);
    CHECK(st.bytes_delivered == 30 * 4096 + 10 * 777);

    // Query (slot 11) after completion: the queue is empty.
    const uint64_t qinfo = com::guest_alloc(sizeof(ga::DStorageQueueInfo));
    fx.vcall(queue, 11, {I(qinfo)});
    ga::DStorageQueueInfo info{};
    CHECK(com::read_guest(qinfo, info));
    CHECK(info.desc.capacity == 127 && info.empty_slot_count == 127);

    // Second batch on the same status entry: the entry goes back to pending at EnqueueStatus.
    fx.enqueue(queue, file_request(file, 123, 1000, dest, 1000));
    fx.vcall(queue, 4, {I(status), I(0)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 0));
    CHECK(std::memcmp(guest_bytes(dest, 1000).data(), data.data() + 123, 1000) == 0);

    fx.vcall(queue, 8);  // Close, then the title's teardown releases the status array only
    CHECK(fx.vcall(status, 2) == 0);
    CHECK(fx.vcall(queue, 2) == 0);
    CHECK(fx.vcall(file, 2) == 0);
    CHECK(fx.vcall(factory, 2) == 1);
}

void test_compressed_and_memory_sources() {
    Fixture fx;
    std::vector<uint8_t> blob(100, 0xEE);
    blob.insert(blob.end(), std::begin(kZlibPage64k), std::end(kZlibPage64k));
    os::scarlett::register_dstorage_virtual_file("G:\\data\\z.ttpf", blob);
    const uint64_t factory = fx.factory();
    const uint64_t file = fx.open_file(factory, u"G:\\data\\z.ttpf");
    const uint64_t queue = fx.create_queue(factory, 64);
    const uint64_t status = fx.create_status(factory, 4);
    const uint64_t dest = com::guest_alloc(65536);

    ga::DStorageRequest r = file_request(file, 100, sizeof(kZlibPage64k), dest, 65536);
    r.options = ga::kDStorageFormatZlib;  // bit 0: the title's hardware-decompressed format
    fx.enqueue(queue, r);
    fx.vcall(queue, 4, {I(status), I(1)});
    // Memory source (bit 22): copy 64 bytes of guest memory.
    const uint64_t src = com::guest_alloc(64);
    std::vector<uint8_t> mem(64);
    for (size_t i = 0; i < mem.size(); ++i) {
        mem[i] = static_cast<uint8_t>(i * 3);
    }
    CHECK(com::write_guest(src, mem.data(), mem.size()));
    const uint64_t dest2 = com::guest_alloc(64);
    ga::DStorageRequest m{};
    m.options = ga::kDStorageOptionSourceMemory;
    m.source = src;
    m.source_size = 64;
    m.destination = dest2;
    m.destination_size = 64;
    fx.enqueue(queue, m);
    // A truncated compressed stream fails with ERROR_INVALID_DATA.
    const uint64_t dest3 = com::guest_alloc(65536);
    ga::DStorageRequest bad = file_request(file, 100, sizeof(kZlibPage64k) - 20, dest3, 65536);
    bad.options = ga::kDStorageFormatZlib;
    fx.enqueue(queue, bad);
    fx.vcall(queue, 4, {I(status), I(2)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 1));
    CHECK(fx.wait_complete(status, 2));
    CHECK(as_hr(fx.vcall(status, 4, {I(1)})) == 0);
    CHECK(as_hr(fx.vcall(status, 4, {I(2)})) == ga::kHrInvalidData);
    CHECK(guest_bytes(dest, 65536) == pattern_data(65536));
    CHECK(guest_bytes(dest2, 64) == mem);
    CHECK(ga::dstorage_stats().decompressed_requests == 1);

    // Unsupported compression format and option bits.
    ga::DStorageRequest f2 = file_request(file, 100, 16, dest, 65536);
    f2.options = 2;
    fx.enqueue(queue, f2);
    fx.vcall(queue, 4, {I(status), I(3)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 3));
    CHECK(as_hr(fx.vcall(status, 4, {I(3)})) == com::E_NOTIMPL_HR);
    ga::DStorageRequest gpu = file_request(file, 0, 16, dest, 65536);
    gpu.options = 1ull << 9;  // a destination-type bit the title never sets
    fx.enqueue(queue, gpu);
    fx.vcall(queue, 4, {I(status), I(3)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 3));
    CHECK(as_hr(fx.vcall(status, 4, {I(3)})) == com::E_NOTIMPL_HR);
}

void test_errors_and_error_record() {
    Fixture fx;
    os::scarlett::register_dstorage_virtual_file("G:\\data\\small.bin", pattern_data(1000));
    const uint64_t factory = fx.factory();
    const uint64_t file = fx.open_file(factory, u"G:\\data\\small.bin");
    const uint64_t queue = fx.create_queue(factory, 64);
    const uint64_t status = fx.create_status(factory, 3);
    const uint64_t dest = com::guest_alloc(4096);
    const uint64_t event = fx.vcall(queue, 9);  // GetErrorEvent
    CHECK(event != 0);
    CHECK(win32::WaitForSingleObject(event, 0) != 0);  // not signaled yet

    // Batch 1: a good read, then a read past the end of the file.
    fx.enqueue(queue, file_request(file, 0, 100, dest, 100));
    fx.enqueue(queue, file_request(file, 900, 200, dest + 512, 200));
    fx.vcall(queue, 4, {I(status), I(0)});
    // Batch 2: only good reads; the failure of batch 1 must not leak into this status entry.
    fx.enqueue(queue, file_request(file, 10, 10, dest + 1024, 10));
    fx.vcall(queue, 4, {I(status), I(1)});
    // Batch 3: a destination that is too small, and a source that is not a file object.
    fx.enqueue(queue, file_request(file, 0, 100, dest, 50));
    fx.enqueue(queue, file_request(status, 0, 10, dest, 10));
    fx.vcall(queue, 4, {I(status), I(2)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 2));
    CHECK(as_hr(fx.vcall(status, 4, {I(0)})) == ga::kHrEof);
    CHECK(as_hr(fx.vcall(status, 4, {I(1)})) == 0);
    CHECK(as_hr(fx.vcall(status, 4, {I(2)})) == com::E_INVALIDARG_HR);
    CHECK(std::memcmp(guest_bytes(dest + 1024, 10).data(), pattern_data(20).data() + 10, 10) == 0);
    CHECK(win32::WaitForSingleObject(event, 0) == 0);  // WAIT_OBJECT_0

    const uint64_t rec_va = com::guest_alloc(sizeof(ga::DStorageErrorRecord));
    fx.vcall(queue, 10, {I(rec_va)});  // RetrieveErrorRecord
    ga::DStorageErrorRecord rec{};
    CHECK(com::read_guest(rec_va, rec));
    CHECK(rec.failure_count == 3);
    CHECK(rec.hresult == ga::kHrEof);
    CHECK(rec.command_type == ga::kDStorageCommandRequest);
    CHECK(rec.request.source_offset == 900 && rec.request.source_size == 200);
    CHECK(std::u16string(rec.filename) == u"G:\\data\\small.bin");
    // Retrieval clears the record and resets the event.
    fx.vcall(queue, 10, {I(rec_va)});
    CHECK(com::read_guest(rec_va, rec));
    CHECK(rec.failure_count == 0);
    CHECK(win32::WaitForSingleObject(event, 0) != 0);
    CHECK(ga::dstorage_stats().requests_failed == 3);
}

void test_cancel_close_and_signal() {
    Fixture fx;
    const std::vector<uint8_t> data = pattern_data(8192);
    os::scarlett::register_dstorage_virtual_file("G:\\data\\c.bin", data);
    const uint64_t factory = fx.factory();
    const uint64_t file = fx.open_file(factory, u"G:\\data\\c.bin");
    const uint64_t queue = fx.create_queue(factory, 64);
    const uint64_t status = fx.create_status(factory, 2);
    const uint64_t dest = com::guest_alloc(8192);
    std::vector<uint8_t> zeros(8192, 0);

    // Tags 0x10 and 0x11 are cancelled by (mask 0xF0, value 0x10); tag 0x20 is not.
    fx.enqueue(queue, file_request(file, 0, 1000, dest, 1000, 0x10));
    fx.enqueue(queue, file_request(file, 1000, 1000, dest + 1000, 1000, 0x11));
    fx.enqueue(queue, file_request(file, 2000, 1000, dest + 2000, 1000, 0x20));
    fx.vcall(queue, 7, {I(0xF0), I(0x10)});  // CancelRequestsWithTag
    const win32::Handle event = win32::CreateEventW(nullptr, true, false, nullptr);
    fx.vcall(queue, 5, {I(event), I(1)});  // EnqueueSignal with a Win32 event
    fx.vcall(queue, 4, {I(status), I(0)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 0));
    CHECK(as_hr(fx.vcall(status, 4, {I(0)})) == 0);
    CHECK(win32::WaitForSingleObject(event, 5000) == 0);
    CHECK(guest_bytes(dest, 2000) == std::vector<uint8_t>(2000, 0));
    CHECK(std::memcmp(guest_bytes(dest + 2000, 1000).data(), data.data() + 2000, 1000) == 0);
    CHECK(ga::dstorage_stats().requests_cancelled == 2);
    win32::CloseHandle(event);

    // Close drops unsubmitted entries; their status entry completes with E_ABORT.
    fx.enqueue(queue, file_request(file, 0, 100, dest + 4000, 100));
    fx.vcall(queue, 4, {I(status), I(1)});
    fx.vcall(queue, 8);  // Close
    CHECK((fx.vcall(status, 3, {I(1)}) & 0xFF) != 0);
    CHECK(as_hr(fx.vcall(status, 4, {I(1)})) == ga::kHrAbort);
    CHECK(guest_bytes(dest + 4000, 100) == std::vector<uint8_t>(100, 0));
    // Enqueues after Close are refused (status entries abort instead of hanging).
    fx.vcall(queue, 4, {I(status), I(1)});
    CHECK(as_hr(fx.vcall(status, 4, {I(1)})) == ga::kHrAbort);
    CHECK(diag_contains("enqueue on a closed queue"));
}

void test_backpressure_and_leaks() {
    Fixture fx;
    const com::BridgeStats before = com::bridge_stats();
    const std::vector<uint8_t> data = pattern_data(1 << 20);
    os::scarlett::register_dstorage_virtual_file("G:\\data\\big.bin", data);
    ga::set_dstorage_worker_threads(2);
    const uint64_t factory = fx.factory();
    const uint64_t file = fx.open_file(factory, u"G:\\data\\big.bin");
    const uint64_t queue = fx.create_queue(factory, 8);  // tiny queue: enqueues must block / auto-submit
    const uint64_t status = fx.create_status(factory, 1);
    const uint64_t dest = com::guest_alloc(1 << 20);
    for (int i = 0; i < 1000; ++i) {
        const uint64_t off = static_cast<uint64_t>((i * 7919) % 1000) * 1000;
        fx.enqueue(queue, file_request(file, off, 1000, dest + off, 1000));
    }
    fx.vcall(queue, 4, {I(status), I(0)});
    fx.vcall(queue, 6);
    CHECK(fx.wait_complete(status, 0));
    CHECK(as_hr(fx.vcall(status, 4, {I(0)})) == 0);
    CHECK(std::memcmp(guest_bytes(dest, 1000000).data(), data.data(), 1000000) == 0);
    const ga::DStorageStats st = ga::dstorage_stats();
    CHECK(st.requests_completed == 1000);
    CHECK(st.submits > 100);  // auto-submits at half capacity

    CHECK(fx.vcall(status, 2) == 0);
    CHECK(fx.vcall(queue, 2) == 0);
    CHECK(fx.vcall(file, 2) == 0);
    CHECK(fx.vcall(factory, 2) == 1);
    ga::reset_xbox_scarlett_guest();  // drops the module's factory reference
    const com::BridgeStats after = com::bridge_stats();
    CHECK(after.live_objects == before.live_objects);
    CHECK(ga::dstorage_stats().live_queues == 0);
}

// Several guest threads feeding several queues at once (run under TSan in CI-like Docker runs).
void test_concurrent_queues() {
    Fixture fx;
    const std::vector<uint8_t> data = pattern_data(1 << 18);
    os::scarlett::register_dstorage_virtual_file("G:\\data\\mt.bin", data);
    ga::set_dstorage_worker_threads(3);
    const uint64_t factory = fx.factory();
    const uint64_t file = fx.open_file(factory, u"G:\\data\\mt.bin");
    constexpr int kThreads = 4;
    constexpr int kRequests = 300;
    uint64_t queues[kThreads];
    uint64_t statuses[kThreads];
    uint64_t dests[kThreads];
    for (int t = 0; t < kThreads; ++t) {
        queues[t] = fx.create_queue(factory, 32);
        statuses[t] = fx.create_status(factory, 1);
        dests[t] = com::guest_alloc(1 << 18);
    }
    std::vector<std::thread> threads;
    std::atomic<int> ok{0};
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            GuestThread gt{};
            if (!create_guest_thread(fx.arena, 256 * 1024, gt)) {
                return;
            }
            for (int i = 0; i < kRequests; ++i) {
                const uint64_t off = static_cast<uint64_t>((i * 37 + t * 11) % 256) * 1024;
                ga::DStorageRequest r = file_request(file, off, 1024, dests[t] + off, 1024);
                const uint64_t va = com::guest_alloc(sizeof(r));
                com::write_guest(va, r);
                fx.vcall_on(gt, queues[t], 3, {I(va)});
                com::guest_free(va);
            }
            fx.vcall_on(gt, queues[t], 4, {I(statuses[t]), I(0)});
            fx.vcall_on(gt, queues[t], 6, {});
            for (int spin = 0; spin < 20000; ++spin) {
                if ((fx.vcall_on(gt, statuses[t], 3, {I(0)}) & 0xFF) != 0) {
                    if (as_hr(fx.vcall_on(gt, statuses[t], 4, {I(0)})) == 0) {
                        ok.fetch_add(1);
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            destroy_guest_thread(fx.arena, gt);
        });
    }
    for (std::thread& th : threads) {
        th.join();
    }
    CHECK(ok.load() == kThreads);
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kRequests; ++i) {
            const uint64_t off = static_cast<uint64_t>((i * 37 + t * 11) % 256) * 1024;
            CHECK(std::memcmp(guest_bytes(dests[t] + off, 1024).data(), data.data() + off, 1024) == 0);
        }
    }
    CHECK(ga::dstorage_stats().requests_completed == static_cast<uint64_t>(kThreads * kRequests));
}

// ---------------------------------------------------------------------------
// GameInput
// ---------------------------------------------------------------------------

struct InputFixture : Fixture {
    InputState shared{};

    InputFixture() { os::scarlett::set_game_input_state(&shared); }
    ~InputFixture() { os::scarlett::set_game_input_state(nullptr); }

    void set_input(uint32_t buttons, float lx = 0.0f, float ly = 0.0f, float lt = 0.0f) {
        InputState v{};
        v.buttons = buttons;
        v.left_stick_x = lx;
        v.left_stick_y = ly;
        v.left_trigger = lt;
        os::GameInputReader::write_monotonic(shared, v);
    }

    uint64_t game_input() {
        const uint64_t out = out_slot();
        CHECK(as_hr(dcall(export_address("GameInput.dll", "GameInputCreate"), {I(out)})) == 0);
        return load64(out);
    }

    uint64_t reading(uint64_t gi, uint32_t kind, uint64_t device = 0, HResult* hr_out = nullptr) {
        const uint64_t out = out_slot();
        const HResult hr = as_hr(vcall(gi, 4, {I(kind), I(device), I(out)}));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        return load64(out);
    }

    uint64_t neighbour(uint64_t gi, uint32_t slot, uint64_t ref, uint32_t kind, HResult* hr_out) {
        const uint64_t out = out_slot();
        *hr_out = as_hr(vcall(gi, slot, {I(ref), I(kind), I(0), I(out)}));
        return load64(out);
    }

    ga::GameInputGamepadState gamepad(uint64_t r, bool* ok) {
        const uint64_t va = com::guest_alloc(sizeof(ga::GameInputGamepadState));
        *ok = (vcall(r, 22, {I(va)}) & 0xFF) != 0;
        ga::GameInputGamepadState s{};
        CHECK(com::read_guest(va, s));
        return s;
    }
};

void test_game_input_readings() {
    InputFixture fx;
    const uint64_t gi = fx.game_input();
    CHECK(gi != 0 && gi == fx.game_input());  // one IGameInput
    CHECK(fx.vcall(gi, 2) == 2);

    fx.set_input(os::input_button::kGamepadA | os::input_button::kRightBumper, 0.0f, 0.0f, 0.5f);
    HResult hr = 0;
    const uint64_t r1 = fx.reading(gi, ga::kGameInputKindGamepad, 0, &hr);
    CHECK(hr == 0 && r1 != 0);
    CHECK(static_cast<uint32_t>(fx.vcall(r1, 3)) == ga::kGameInputKindGamepad);  // GetInputKind
    CHECK(fx.vcall(r1, 5) != 0);                                                 // GetTimestamp
    bool ok = false;
    ga::GameInputGamepadState s = fx.gamepad(r1, &ok);
    CHECK(ok);
    CHECK(s.buttons == (0x4u | 0x800u));  // A | RightShoulder
    CHECK(s.left_trigger > 0.4f && s.left_trigger < 0.6f);
    CHECK(fx.vcall(r1, 14) == 0);  // a gamepad reading has no keys
    const uint64_t seq1 = fx.vcall(r1, 4, {I(ga::kGameInputKindGamepad)});

    // No new input: GetNextReading finds nothing; the current reading is the same snapshot.
    CHECK(fx.neighbour(gi, 5, r1, ga::kGameInputKindGamepad, &hr) == 0);
    CHECK(hr == ga::kGameInputReadingNotFound);
    const uint64_t same = fx.reading(gi, ga::kGameInputKindGamepad);
    CHECK(fx.vcall(same, 4, {I(0)}) == seq1);
    CHECK(fx.vcall(same, 2) == 0);

    // New input: next reading, then walking back with GetPreviousReading.
    fx.set_input(os::input_button::kGamepadB, 0.0f, 0.0f, 0.0f);
    const uint64_t r2 = fx.neighbour(gi, 5, r1, ga::kGameInputKindGamepad, &hr);
    CHECK(hr == 0 && r2 != 0);
    CHECK(fx.vcall(r2, 4, {I(0)}) > seq1);
    s = fx.gamepad(r2, &ok);
    CHECK(ok && s.buttons == 0x8u);
    const uint64_t back = fx.neighbour(gi, 6, r2, ga::kGameInputKindGamepad, &hr);
    CHECK(hr == 0 && fx.vcall(back, 4, {I(0)}) == seq1);
    CHECK(fx.neighbour(gi, 6, r1, ga::kGameInputKindGamepad, &hr) == 0);
    CHECK(hr == ga::kGameInputReadingNotFound);

    // Keyboard readings carry the SAB keys (W and Space).
    fx.set_input(os::input_button::kMoveForward | os::input_button::kJump);
    const uint64_t kb = fx.reading(gi, ga::kGameInputKindKeyboard, 0, &hr);
    CHECK(hr == 0);
    CHECK(fx.vcall(kb, 14) == 2);
    const uint64_t keys = com::guest_alloc(16 * sizeof(ga::GameInputKeyState));
    CHECK(fx.vcall(kb, 15, {I(16), I(keys)}) == 2);
    ga::GameInputKeyState k[2]{};
    CHECK(com::read_guest(keys, k, sizeof(k)));
    CHECK(k[0].scan_code == 0x11 && k[0].virtual_key == 0x57 && k[0].code_point == 'w');
    CHECK(k[1].scan_code == 0x39 && k[1].virtual_key == 0x20);
    CHECK(fx.vcall(kb, 15, {I(1), I(keys)}) == 1);
    fx.gamepad(kb, &ok);
    CHECK(!ok);  // not a gamepad reading
    // Mouse readings are not provided.
    CHECK(fx.reading(gi, ga::kGameInputKindMouse, 0, &hr) == 0);
    CHECK(hr == ga::kGameInputReadingNotFound);

    // GetDevice on a reading returns an AddRef'd device.
    const uint64_t dev_out = fx.out_slot();
    fx.vcall(r1, 6, {I(dev_out)});
    const uint64_t device = load64(dev_out);
    CHECK(device != 0);
    CHECK(static_cast<uint32_t>(fx.vcall(device, 4)) == (ga::kGameInputDeviceConnected | ga::kGameInputDeviceInputEnabled));
    CHECK(fx.vcall(device, 2) >= 1);

    // Every-frame pattern: GetCurrentReading + Release leaks nothing.
    for (uint64_t r : {r1, r2, back, kb}) {
        CHECK(fx.vcall(r, 2) == 0);
    }
    const uint64_t frame_out = fx.out_slot();
    const com::BridgeStats before = com::bridge_stats();
    for (int i = 0; i < 200; ++i) {
        fx.set_input(static_cast<uint32_t>(i & 1) * os::input_button::kGamepadX);
        CHECK(as_hr(fx.vcall(gi, 4, {I(ga::kGameInputKindGamepad), I(0), I(frame_out)})) == 0);
        CHECK(fx.vcall(load64(frame_out), 2) == 0);
    }
    const com::BridgeStats after = com::bridge_stats();
    CHECK(after.live_objects == before.live_objects);
    CHECK(after.guest_bytes_in_use == before.guest_bytes_in_use);

    // Unused slots are named logging stubs.
    bool faulted = false;
    try {
        fx.vcall(gi, 7, {I(0), I(0), I(0)});
    } catch (const GuestFault&) {
        faulted = true;
    }
    CHECK(faulted);
    CHECK(diag_contains("GameInput.IGameInput"));
    CHECK(diag_contains("(GetTemporalReading)"));
    bool reported = false;
    for (const com::StubHit& h : com::stub_hit_report()) {
        reported |= h.interface_name == "GameInput.IGameInput" && h.slot == 7 && h.hits == 1;
    }
    CHECK(reported);
}

void test_game_input_devices() {
    InputFixture fx;
    const uint64_t gi = fx.game_input();
    g_device_cb.calls = 0;
    const uint64_t token_va = fx.out_slot();
    // The title's gamepad registration: no device, Gamepad kind, Connected filter, blocking.
    CHECK(as_hr(fx.vcall(gi, 9, {I(0), I(ga::kGameInputKindGamepad), I(1), I(2), I(0x1234), I(kGuestDeviceCallback),
                                 I(token_va)})) == 0);
    CHECK(g_device_cb.calls.load() == 1);
    CHECK(g_device_cb.context == 0x1234);
    CHECK(g_device_cb.token == load64(token_va) && g_device_cb.token != 0);
    CHECK((g_device_cb.current & ga::kGameInputDeviceConnected) != 0);
    CHECK(g_device_cb.previous == 0);
    const uint64_t pad = g_device_cb.device;
    CHECK(pad != 0);

    // GetDeviceInfo -> deviceId at +0x20 -> FindDeviceFromId finds the same device.
    const uint64_t info = fx.vcall(pad, 3);
    CHECK(info != 0);
    ga::AppLocalDeviceId id{};
    CHECK(com::read_guest(info + ga::kDeviceInfoDeviceId, id));
    uint32_t supported = 0;
    CHECK(com::read_guest(info + ga::kDeviceInfoSupportedInput, supported));
    CHECK(supported == ga::kGameInputKindGamepad);
    const uint64_t id_va = com::guest_alloc(sizeof(id));
    CHECK(com::write_guest(id_va, id));
    const uint64_t found = fx.out_slot();
    CHECK(as_hr(fx.vcall(gi, 16, {I(id_va), I(found)})) == 0);
    CHECK(load64(found) == pad);
    CHECK(fx.vcall(pad, 2) >= 1);  // FindDeviceFromId's reference
    id.value[0] ^= 0xFF;
    CHECK(com::write_guest(id_va, id));
    CHECK(as_hr(fx.vcall(gi, 16, {I(id_va), I(found)})) == ga::kGameInputDeviceNotFound);
    CHECK(load64(found) == 0);

    // Readings filtered by the device; a keyboard device does not give gamepad readings.
    HResult hr = 0;
    const uint64_t r = fx.reading(gi, ga::kGameInputKindGamepad, pad, &hr);
    CHECK(hr == 0 && r != 0);
    CHECK(fx.vcall(r, 2) == 0);

    // SetRumbleState (slot 10)
    const ga::GameInputRumbleParams rumble{0.25f, 0.5f, 0.0f, 1.0f};
    const uint64_t rumble_va = com::guest_alloc(sizeof(rumble));
    CHECK(com::write_guest(rumble_va, rumble));
    fx.vcall(pad, 10, {I(rumble_va)});
    const ga::GameInputRumbleParams got = ga::virtual_gamepad_rumble();
    CHECK(got.low_frequency == 0.25f && got.high_frequency == 0.5f && got.right_trigger == 1.0f);

    // The keyboard registration enumerates the keyboard device; NoEnumeration calls nothing.
    const uint64_t token2 = fx.out_slot();
    CHECK(as_hr(fx.vcall(gi, 9, {I(0), I(ga::kGameInputKindKeyboard), I(1), I(2), I(0), I(kGuestDeviceCallback),
                                 I(token2)})) == 0);
    CHECK(g_device_cb.calls.load() == 2);
    CHECK(g_device_cb.device != pad);
    const uint64_t keyboard = g_device_cb.device;
    CHECK(as_hr(fx.vcall(gi, 9, {I(0), I(ga::kGameInputKindGamepad), I(1), I(0), I(0), I(kGuestDeviceCallback),
                                 I(fx.out_slot())})) == 0);
    CHECK(g_device_cb.calls.load() == 2);
    HResult khr = 0;
    CHECK(fx.reading(gi, ga::kGameInputKindGamepad, keyboard, &khr) == 0);
    CHECK(khr == ga::kGameInputReadingNotFound);

    // UnregisterCallback (slot 13) -> bool
    CHECK((fx.vcall(gi, 13, {I(load64(token_va)), I(5000)}) & 0xFF) == 1);
    CHECK((fx.vcall(gi, 13, {I(load64(token_va)), I(5000)}) & 0xFF) == 0);
    CHECK((fx.vcall(gi, 13, {I(load64(token2)), I(~0ull)}) & 0xFF) == 1);

    // Releasing the guest's references (the module keeps one) destroys nothing yet; dropping the
    // module reference destroys IGameInput and both devices.
    CHECK(fx.vcall(gi, 2) == 1);
    const uint64_t live_before = com::bridge_stats().live_objects;
    ga::reset_xbox_scarlett_guest();
    CHECK(com::bridge_stats().live_objects == live_before - 3);
}

// ---------------------------------------------------------------------------
// xg_xs
// ---------------------------------------------------------------------------

uint64_t xg_desc(uint32_t dimension, uint64_t width, uint32_t height, uint16_t depth, uint16_t mips, uint32_t format) {
    ga::XgResourceDesc d{};
    d.dimension = dimension;
    d.width = width;
    d.height = height;
    d.depth_or_array_size = depth;
    d.mip_levels = mips;
    d.format = format;
    d.sample_count = 1;
    d.swizzle_mode = 0x100;
    const uint64_t va = com::guest_alloc(0x40);
    CHECK(com::write_guest(va, &d, ga::kXgResourceDescBytes));
    return va;
}

void test_xg() {
    Fixture fx;
    const uint64_t create = fx.export_address("xg_xs.dll", "XGCreateTextureComputer");
    const uint64_t suggest = fx.export_address("xg_xs.dll", "XGSuggestSwizzleMode");

    // XGSuggestSwizzleMode(desc, 0, &desc.mode)
    const uint64_t desc = xg_desc(3, 256, 256, 1, 0, 28);
    CHECK(as_hr(fx.dcall(suggest, {I(desc), I(0), I(desc + 0x2C)})) == 0);
    uint32_t mode = 0;
    CHECK(com::read_guest(desc + 0x2C, mode));
    CHECK(mode == ga::kXgDefaultSwizzleMode);

    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(create, {I(desc), I(out)})) == 0);
    const uint64_t comp = load64(out);
    CHECK(comp != 0);

    const uint64_t layout_va = com::guest_alloc(ga::kXgResourceLayoutSize);
    CHECK(as_hr(fx.vcall(comp, 2, {I(layout_va)})) == 0);
    const std::vector<uint8_t> layout = guest_bytes(layout_va, ga::kXgResourceLayoutSize);
    uint64_t total = 0;
    uint32_t mips = 0;
    std::memcpy(&total, layout.data(), 8);
    std::memcpy(&mips, layout.data() + 0x10, 4);
    CHECK(mips == 9);
    ga::XgMipLayout m0{};
    ga::XgMipLayout m1{};
    std::memcpy(&m0, layout.data() + ga::kXgLayoutPlaneOffset + ga::kXgPlaneMipOffset, sizeof(m0));
    std::memcpy(&m1, layout.data() + ga::kXgLayoutPlaneOffset + ga::kXgPlaneMipOffset + ga::kXgMipLayoutSize, sizeof(m1));
    // Linear (0x100): addrlib ADDR_SW_LINEAR, mips smallest-first with 256 B-aligned pitches, so
    // mip 0 sits after mips 8..1 (98,048 B) and mip 1 after mips 8..2 (32,512 B).
    CHECK(m0.pitch_bytes == 1024 && m0.size_bytes == 256u * 1024u && m0.slice_2d_size_bytes == 256u * 1024u);
    CHECK(m0.offset_bytes == 98048u && m0.width_elements == 256 && m0.height_elements == 256);
    CHECK(m1.pitch_bytes == 512 && m1.offset_bytes == 32512u);
    CHECK(total == 98048u + 256u * 1024u);

    // CopyIntoSubresource (slot 8) of mip 0 from a tightly packed source.
    std::vector<uint8_t> texels(256 * 256 * 4);
    for (size_t i = 0; i < texels.size(); ++i) {
        texels[i] = static_cast<uint8_t>(i * 13 + (i >> 10));
    }
    const uint64_t src = com::guest_alloc(texels.size());
    CHECK(com::write_guest(src, texels.data(), texels.size()));
    const uint64_t dst = com::guest_alloc(total);
    CHECK(as_hr(fx.vcall(comp, 8, {I(dst), I(0), I(0), I(src), I(1024), I(256 * 1024)})) == 0);
    CHECK(guest_bytes(dst + 98048u, texels.size()) == texels);
    CHECK(as_hr(fx.vcall(comp, 8, {I(0), I(0), I(0), I(src), I(1024), I(0)})) == com::E_POINTER_HR);

    // Slot 1 drops the computer (no IUnknown on this interface).
    CHECK(fx.vcall(comp, 0) == 2);
    CHECK(fx.vcall(comp, 1) == 1);
    CHECK(fx.vcall(comp, 1) == 0);

    // GetResourceTiling (slot 10) for a BC1 4096x4096 texture with a full mip chain.
    const uint64_t out2 = fx.out_slot();
    CHECK(as_hr(fx.dcall(create, {I(xg_desc(3, 4096, 4096, 1, 0, 71)), I(out2)})) == 0);
    const uint64_t bc = load64(out2);
    const uint64_t num_tiles = com::guest_alloc(4);
    const uint64_t packed = com::guest_alloc(12);
    const uint64_t shape = com::guest_alloc(12);
    const uint64_t count = com::guest_alloc(4);
    const uint64_t tilings = com::guest_alloc(16 * 12);
    uint32_t cap = 16;
    CHECK(com::write_guest(count, cap));
    fx.vcall(bc, 10, {I(num_tiles), I(packed), I(shape), I(count), I(0), I(tilings)});
    ga::XgTileShape ts{};
    ga::XgPackedMipInfo pm{};
    CHECK(com::read_guest(shape, ts));
    CHECK(com::read_guest(packed, pm));
    CHECK(ts.width_in_texels == 512 && ts.height_in_texels == 256 && ts.depth_in_texels == 1);
    CHECK(pm.num_standard_mips == 4 && pm.num_packed_mips == 9);
    CHECK(pm.num_tiles_for_packed_mips >= 1);
    uint32_t written = 0;
    CHECK(com::read_guest(count, written));
    CHECK(written == 13);
    ga::XgSubresourceTiling t0{};
    ga::XgSubresourceTiling t3{};
    ga::XgSubresourceTiling t4{};
    CHECK(com::read_guest(tilings, t0));
    CHECK(com::read_guest(tilings + 36, t3));
    CHECK(com::read_guest(tilings + 48, t4));
    CHECK(t0.width_in_tiles == 8 && t0.height_in_tiles == 16 && t0.start_tile_index_in_overall_resource == 0);
    CHECK(t3.width_in_tiles == 1 && t3.height_in_tiles == 2);
    CHECK(t4.start_tile_index_in_overall_resource == ga::kXgPackedTile);
    uint32_t tiles = 0;
    CHECK(com::read_guest(num_tiles, tiles));
    CHECK(tiles == 128 + 32 + 8 + 2 + pm.num_tiles_for_packed_mips);
    CHECK(pm.start_tile_index_in_overall_resource == 170);
    CHECK(fx.vcall(bc, 1) == 0);

    // GetResourceTiling for 64KB_D (0x10A): tiles in GFX10 memory order. A 512x512 BC7 array of
    // 32 with 10 mips: per slice the mip tail (mips 2..9) is tile 0, mip 1 tile 1 and mip 0
    // (2x2 tiles) starts at tile 2; 6 tiles per slice.
    {
        const uint64_t tdesc = xg_desc(3, 512, 512, 32, 10, 98);
        CHECK(com::write_guest(tdesc + 0x2C, uint32_t{0x10A}));
        const uint64_t tout = fx.out_slot();
        CHECK(as_hr(fx.dcall(create, {I(tdesc), I(tout)})) == 0);
        const uint64_t t64 = load64(tout);
        uint32_t tcap = 16;
        CHECK(com::write_guest(count, tcap));
        fx.vcall(t64, 10, {I(num_tiles), I(packed), I(shape), I(count), I(0), I(tilings)});
        ga::XgPackedMipInfo tp{};
        CHECK(com::read_guest(packed, tp));
        CHECK(tp.num_standard_mips == 2 && tp.num_packed_mips == 8);
        CHECK(tp.num_tiles_for_packed_mips == 1 && tp.start_tile_index_in_overall_resource == 0);
        uint32_t ttiles = 0;
        CHECK(com::read_guest(num_tiles, ttiles));
        CHECK(ttiles == 6u * 32u);
        ga::XgSubresourceTiling m0{};
        ga::XgSubresourceTiling m1{};
        ga::XgSubresourceTiling m2{};
        ga::XgSubresourceTiling s1m0{};
        CHECK(com::read_guest(tilings, m0));
        CHECK(com::read_guest(tilings + 12, m1));
        CHECK(com::read_guest(tilings + 24, m2));
        CHECK(m0.width_in_tiles == 2 && m0.height_in_tiles == 2 && m0.start_tile_index_in_overall_resource == 2);
        CHECK(m1.width_in_tiles == 1 && m1.height_in_tiles == 1 && m1.start_tile_index_in_overall_resource == 1);
        CHECK(m2.start_tile_index_in_overall_resource == ga::kXgPackedTile);
        // Subresource 10 = slice 1, mip 0: one slice (6 tiles) further.
        uint32_t one = 1;
        CHECK(com::write_guest(count, one));
        fx.vcall(t64, 10, {I(num_tiles), I(packed), I(shape), I(count), I(10), I(tilings)});
        CHECK(com::read_guest(tilings, s1m0));
        CHECK(s1m0.start_tile_index_in_overall_resource == 8);
        CHECK(fx.vcall(t64, 1) == 0);
    }

    // GetTexelElementOffsetBytes (slot 6): the GFX10 address of an element (the detiler's), plus
    // the slice's chain; bit offset 0; invalid plane / mip / coordinate -> UINT64_MAX.
    {
        namespace gx = playgta6::gpu::gfx10;
        const uint64_t tdesc = xg_desc(3, 512, 512, 2, 10, 98);
        CHECK(com::write_guest(tdesc + 0x2C, uint32_t{0x10A}));
        const uint64_t tout = fx.out_slot();
        CHECK(as_hr(fx.dcall(create, {I(tdesc), I(tout)})) == 0);
        const uint64_t tc = load64(tout);
        const gx::TextureLayoutDesc chain{98u, 512u, 512u, 1u, 10u, 0x10Au, 0u};
        const uint64_t chain_bytes = gx::surface_bytes(chain);
        const uint64_t bit = com::guest_alloc(4);
        CHECK(com::write_guest(bit, uint32_t{0xFFFFFFFFu}));
        for (const uint32_t mip : {0u, 1u, 3u}) {
            const uint64_t got = fx.vcall(tc, 6, {I(0), I(mip), I(3), I(5), I(1), I(0), I(bit)});
            CHECK(got == chain_bytes + gx::surface_element_offset(chain, mip, 3u, 5u, 0u));
        }
        uint32_t bits = 1;
        CHECK(com::read_guest(bit, bits) && bits == 0u);
        CHECK(fx.vcall(tc, 6, {I(0), I(0), I(0), I(0), I(0), I(0), I(0)}) == gx::surface_element_offset(chain, 0u, 0u, 0u, 0u));
        CHECK(fx.vcall(tc, 6, {I(1), I(0), I(0), I(0), I(0), I(0), I(0)}) == UINT64_MAX);    // plane
        CHECK(fx.vcall(tc, 6, {I(0), I(10), I(0), I(0), I(0), I(0), I(0)}) == UINT64_MAX);   // mip
        CHECK(fx.vcall(tc, 6, {I(0), I(0), I(128), I(0), I(0), I(0), I(0)}) == UINT64_MAX);  // x (128 blocks)
        CHECK(fx.vcall(tc, 6, {I(0), I(0), I(0), I(0), I(2), I(0), I(0)}) == UINT64_MAX);    // slice
        CHECK(fx.vcall(tc, 1) == 0);
    }

    // An unsupported dimension fails.
    CHECK(as_hr(fx.dcall(create, {I(xg_desc(7, 16, 16, 1, 1, 28)), I(fx.out_slot())})) == com::E_INVALIDARG_HR);

    // A tiled swizzle mode (+0x2C = 0x100 | n): GetResourceLayout reports the GFX10 surface and
    // CopyIntoSubresource writes the GFX10 layout that the runtime's detiler reads back, per mip
    // (mip 1 of a 9-mip 256x256 RGBA8 sits where the GFX10 chain puts it), per array slice, for a
    // thick 3D surface and for 64KB_D. An unverified mode (64KB_R_X) keeps the linear answer.
    // Linear 0x100 (#678): the same placement, and the D3D12 allocation size for the same
    // description equals the XG total (one rule for allocation, XG and the upload's de-tiler).
    namespace gfx10 = playgta6::gpu::gfx10;
    namespace dg = playgta6::gpu::d3d12_guest;
    struct TiledCase {
        uint32_t dim, w, h, d, mips, format, bpe, mip, slice, swizzle;
    };
    for (const TiledCase& c : {TiledCase{3, 256, 256, 1, 9, 28, 4, 1, 0, 0x105}, TiledCase{3, 64, 64, 1, 1, 28, 4, 0, 0, 0x105},
                               TiledCase{4, 16, 16, 16, 1, 10, 8, 0, 0, 0x105}, TiledCase{3, 256, 256, 3, 2, 61, 1, 1, 2, 0x10A},
                               TiledCase{3, 512, 256, 1, 3, 71, 8, 0, 0, 0x10A}, TiledCase{3, 128, 72, 1, 7, 41, 4, 3, 0, 0x100},
                               TiledCase{3, 128, 72, 1, 7, 41, 4, 0, 0, 0x100}, TiledCase{3, 20, 6, 3, 3, 28, 4, 1, 2, 0x100},
                               TiledCase{4, 32, 32, 36, 1, 42, 4, 0, 0, 0x100}, TiledCase{3, 100, 60, 1, 5, 71, 8, 2, 0, 0x100},
                               TiledCase{2, 4096, 1, 1, 3, 41, 4, 2, 0, 0x100}}) {
        const uint64_t tdesc = xg_desc(c.dim, c.w, c.h, c.d, c.mips, c.format);
        CHECK(com::write_guest(tdesc + 0x2C, c.swizzle));
        const uint64_t tout = fx.out_slot();
        CHECK(as_hr(fx.dcall(create, {I(tdesc), I(tout)})) == 0);
        const uint64_t tcomp = load64(tout);
        const gfx10::FormatGeometry fg = gfx10::format_geometry(c.format);
        const uint32_t mw = std::max(1u, c.w >> c.mip), mh = std::max(1u, c.h >> c.mip);
        const uint32_t md = c.dim == 4 ? std::max(1u, c.d >> c.mip) : 1u;
        const uint32_t row = (mw + fg.block_width - 1) / fg.block_width * c.bpe;
        const uint32_t rows = (mh + fg.block_height - 1) / fg.block_height;
        std::vector<uint8_t> lin(size_t{row} * rows * md);
        for (size_t i = 0; i < lin.size(); ++i) {
            lin[i] = static_cast<uint8_t>(i * 7 + (i >> 9));
        }
        const uint64_t tsrc = com::guest_alloc(lin.size());
        CHECK(com::write_guest(tsrc, lin.data(), lin.size()));
        const gfx10::TextureLayoutDesc gd{c.format, c.w, c.h, c.dim == 4 ? c.d : 1u, c.mips, c.swizzle, 0};
        const size_t chain = gfx10::surface_bytes(gd);
        const uint32_t slices = c.dim != 4 ? c.d : 1u;
        CHECK(chain != 0);
        dg::GuestResourceDesc rd{};
        rd.Dimension = c.dim;
        rd.Width = c.w;
        rd.Height = c.h;
        rd.DepthOrArraySize = static_cast<uint16_t>(c.d);
        rd.MipLevels = static_cast<uint16_t>(c.mips);
        rd.Format = c.format;
        rd.SampleCount = 1;
        rd.Layout = c.swizzle;
        CHECK(dg::resource_size_estimate(rd) == chain * slices);

        const uint64_t tlayout = com::guest_alloc(ga::kXgResourceLayoutSize);
        CHECK(as_hr(fx.vcall(tcomp, 2, {I(tlayout)})) == 0);
        const std::vector<uint8_t> lb = guest_bytes(tlayout, ga::kXgResourceLayoutSize);
        uint64_t ttotal = 0;
        std::memcpy(&ttotal, lb.data(), 8);
        CHECK(ttotal == chain * slices);
        ga::XgMipLayout tm{};
        std::memcpy(&tm, lb.data() + ga::kXgLayoutPlaneOffset + ga::kXgPlaneMipOffset + c.mip * ga::kXgMipLayoutSize,
                    sizeof(tm));
        const gfx10::MipPlacement mp = gfx10::mip_placement(gd, c.mip);
        CHECK(tm.pitch_bytes == mp.pitch_elements * c.bpe && tm.offset_bytes == mp.offset_bytes);
        CHECK(tm.size_bytes == mp.size_bytes * slices && tm.slice_2d_size_bytes == mp.slice_bytes);

        const uint64_t tdst = com::guest_alloc(chain * slices + 0x10000);
        const uint32_t sub = c.slice * c.mips + c.mip;
        CHECK(as_hr(fx.vcall(tcomp, 8, {I(tdst), I(0), I(sub), I(tsrc), I(row), I(row * rows)})) == 0);
        const std::vector<uint8_t> written = guest_bytes(tdst + c.slice * chain, chain);
        std::vector<uint8_t> back;
        CHECK(gfx10::detile_from_surface(gd, c.mip, written.data(), written.size(), back).ok);
        CHECK(back == lin);
        CHECK(fx.vcall(tcomp, 1) == 0);
    }
    {
        const uint64_t rdesc = xg_desc(3, 256, 256, 1, 1, 28);
        const uint32_t rx = 0x11B;
        CHECK(com::write_guest(rdesc + 0x2C, rx));
        const uint64_t rout = fx.out_slot();
        CHECK(as_hr(fx.dcall(create, {I(rdesc), I(rout)})) == 0);
        const uint64_t rcomp = load64(rout);
        const uint64_t rlayout = com::guest_alloc(ga::kXgResourceLayoutSize);
        CHECK(as_hr(fx.vcall(rcomp, 2, {I(rlayout)})) == 0);
        ga::XgMipLayout rm{};
        std::memcpy(&rm, guest_bytes(rlayout, ga::kXgResourceLayoutSize).data() + ga::kXgLayoutPlaneOffset +
                             ga::kXgPlaneMipOffset, sizeof(rm));
        CHECK(rm.pitch_bytes == 1024 && rm.size_bytes == 256u * 1024u);
        CHECK(fx.vcall(rcomp, 1) == 0);
    }
}

// AcpHalCreate: a guest object without an IUnknown prefix. Connect (0), submit (2), disconnect
// (1), release (5); commands outside a connection fail; slots 3/4 trap.
void test_acp_hal() {
    Fixture fx;
    const uint64_t create = fx.export_address("AcpHal.dll", "AcpHalCreate");
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(create, {I(out)})) == 0);
    const uint64_t hal = load64(out);
    CHECK(hal != 0 && ga::acp_hal_state(hal).valid && !ga::acp_hal_state(hal).connected);

    const uint64_t payload = com::guest_alloc(4);
    CHECK(com::write_guest(payload, uint32_t{0xC0}));
    CHECK(as_hr(fx.vcall(hal, 2, {I(2), I(0xF0F0F0F0u), I(0xFFFFFFFFu), I(payload), I(0)})) < 0);  // not connected
    CHECK(as_hr(fx.vcall(hal, 0, {I(0x1000), I(0x1000)})) == 0);
    CHECK(ga::acp_hal_state(hal).connected);
    CHECK(as_hr(fx.vcall(hal, 2, {I(2), I(0xF0F0F0F0u), I(0xFFFFFFFFu), I(payload), I(0)})) == 0);
    CHECK(as_hr(fx.vcall(hal, 2, {I(5), I(0xF6F6F6F6u), I(0xFFFFFFFFu), I(payload), I(0)})) == 0);
    CHECK(as_hr(fx.vcall(hal, 2, {I(5), I(0xF6F6F6F6u), I(0xFFFFFFFFu), I(payload), I(0)})) == 0);
    CHECK(ga::acp_hal_state(hal).commands_2 == 1 && ga::acp_hal_state(hal).commands_5 == 2);
    CHECK(as_hr(fx.vcall(hal, 0, {I(0), I(0x1000)})) == com::E_INVALIDARG_HR);

    bool faulted = false;
    try {
        fx.vcall(hal, 3);
    } catch (const GuestFault& f) {
        faulted = f.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot);
    }
    CHECK(faulted);

    CHECK(as_hr(fx.vcall(hal, 1)) == 0);
    CHECK(!ga::acp_hal_state(hal).connected);
    CHECK(fx.vcall(hal, 5) == 0);  // the last reference
    CHECK(!ga::acp_hal_state(hal).valid);
    CHECK(as_hr(fx.dcall(create, {I(0)})) == com::E_POINTER_HR);
}

// ---------------------------------------------------------------------------
// MMDevice: the default audio render endpoint
// ---------------------------------------------------------------------------

constexpr Guid kClsidMMDeviceEnumerator = com::make_guid("bcde0395-e52f-467c-8e3d-c4579291692e");
constexpr Guid kIidMMDeviceEnumerator = com::make_guid("a95664d2-9614-4f35-a746-de8db63617e6");
constexpr Guid kIidAudioClient = com::make_guid("1cb9ad4c-dbfa-4c32-b178-c2f568a703b2");
constexpr Guid kIidAudioRenderClient = com::make_guid("f294acfc-3146-4483-a7bf-addca7c260e2");
constexpr Guid kIidAudioClock = com::make_guid("cd63314f-3fba-4a1b-812c-ef96358728e7");
constexpr Guid kIidSpatialAudioClient = com::make_guid("bbf8e066-aaaa-49be-9a4d-fd2a858ea27f");
constexpr Guid kIidSpatialStream = com::make_guid("bab5f473-b423-477b-85f5-b5a332a04153");
constexpr Guid kPkeyFormFactor = com::make_guid("1da5d803-d492-4edd-8c23-e0c0ffee7f0e");
constexpr Guid kPkeyFriendlyName = com::make_guid("a45c254e-df1c-4efd-8020-67d146a850e0");

// The audio objects hand out CoTaskMem strings and formats from the win32 process heap, which
// binds to the arena active at its first use for the rest of the process: the audio cases share
// one fixture whose arena the process heap is pointed at. The 10 ms period clock runs by hand.
struct AudioFixture : Fixture {
    AudioFixture() {
        win32::set_vmem_arena(&arena);
        ga::audio_engine_manual_clock(true);
    }
    ~AudioFixture() {
        ga::reset_audio_endpoint();
        win32::set_vmem_arena(nullptr);
    }
    void next_case() {
        ga::reset_audio_endpoint();
        ga::audio_engine_manual_clock(true);
    }

    uint64_t enumerator() {
        const uint64_t create = export_address("combase.dll", "CoCreateInstance");
        const uint64_t out = out_slot();
        CHECK(as_hr(dcall(create, {I(guid(kClsidMMDeviceEnumerator)), I(0), I(0x17),
                                   I(guid(kIidMMDeviceEnumerator)), I(out)})) == 0);
        return load64(out);
    }
    uint64_t default_endpoint(uint64_t enumerator) {
        const uint64_t out = out_slot();
        CHECK(as_hr(vcall(enumerator, 4, {I(0), I(0), I(out)})) == 0);  // eRender, eConsole
        return load64(out);
    }
    uint64_t activate(uint64_t device, const Guid& iid) {
        const uint64_t out = out_slot();
        CHECK(as_hr(vcall(device, 3, {I(guid(iid)), I(1), I(0), I(out)})) == 0);
        return load64(out);
    }
    std::u16string wide(uint64_t va) {
        std::u16string s;
        for (char16_t c = 0; va != 0 && com::read_guest(va + 2 * s.size(), c) && c != 0;) {
            s.push_back(c);
        }
        return s;
    }
};

// CoCreateInstance(CLSID_MMDeviceEnumerator) -> one active render endpoint, reachable through the
// collection, the default endpoint and its id; its property store has the form factor and name.
void mmdevice_enumerator_and_endpoint(AudioFixture& fx) {
    const uint64_t create = fx.export_address("combase.dll", "CoCreateInstance");
    const uint64_t out = fx.out_slot();
    const Guid unknown = com::make_guid("01234567-89ab-cdef-0123-456789abcdef");
    CHECK(as_hr(fx.dcall(create, {I(fx.guid(unknown)), I(0), I(1), I(fx.guid(kIidMMDeviceEnumerator)), I(out)})) ==
          static_cast<HResult>(0x80040154u));  // REGDB_E_CLASSNOTREG
    CHECK(load64(out) == 0);

    const uint64_t en = fx.enumerator();
    CHECK(en != 0);
    CHECK(as_hr(fx.vcall(en, 6, {I(com::guest_alloc(8))})) == 0);  // RegisterEndpointNotificationCallback

    // EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE): one endpoint; eCapture: none.
    CHECK(as_hr(fx.vcall(en, 3, {I(0), I(1), I(out)})) == 0);
    const uint64_t coll = load64(out);
    const uint64_t count = com::guest_alloc(4);
    CHECK(as_hr(fx.vcall(coll, 3, {I(count)})) == 0 && load32(count) == 1);
    CHECK(as_hr(fx.vcall(coll, 4, {I(1), I(out)})) == com::E_INVALIDARG_HR && load64(out) == 0);
    CHECK(as_hr(fx.vcall(coll, 4, {I(0), I(out)})) == 0);
    const uint64_t item = load64(out);
    CHECK(item != 0);
    CHECK(as_hr(fx.vcall(en, 3, {I(1), I(1), I(out)})) == 0);
    const uint64_t capture = load64(out);
    CHECK(as_hr(fx.vcall(capture, 3, {I(count)})) == 0 && load32(count) == 0);
    CHECK(as_hr(fx.vcall(en, 4, {I(1), I(0), I(out)})) == static_cast<HResult>(0x80070490u));  // no capture

    const uint64_t dev = fx.default_endpoint(en);
    CHECK(as_hr(fx.vcall(dev, 6, {I(count)})) == 0 && load32(count) == 1);  // DEVICE_STATE_ACTIVE
    CHECK(as_hr(fx.vcall(dev, 5, {I(out)})) == 0);  // GetId
    const uint64_t id = load64(out);
    CHECK(!fx.wide(id).empty());
    CHECK(as_hr(fx.vcall(item, 5, {I(out)})) == 0 && fx.wide(load64(out)) == fx.wide(id));
    pg6_win32_co_task_mem_free(load64(out));
    CHECK(as_hr(fx.vcall(en, 5, {I(id), I(out)})) == 0 && load64(out) != 0);  // GetDevice(id)
    CHECK(as_hr(fx.vcall(en, 5, {I(fx.wstr(u"{nope}")), I(out)})) == static_cast<HResult>(0x80070490u));
    pg6_win32_co_task_mem_free(id);

    // OpenPropertyStore(STGM_READ): FormFactor = Speakers (VT_UI4), FriendlyName (VT_LPWSTR).
    CHECK(as_hr(fx.vcall(dev, 4, {I(0), I(out)})) == 0);
    const uint64_t store = load64(out);
    const uint64_t key = com::guest_alloc(20);
    const uint64_t pv = com::guest_alloc(24);
    CHECK(com::write_guest(key, kPkeyFormFactor) && com::write_guest(key + 16, uint32_t{0}));
    CHECK(as_hr(fx.vcall(store, 5, {I(key), I(pv)})) == 0);
    CHECK((load32(pv) & 0xFFFF) == 19 && load32(pv + 8) == 1);
    CHECK(com::write_guest(key, kPkeyFriendlyName) && com::write_guest(key + 16, uint32_t{14}));
    CHECK(as_hr(fx.vcall(store, 5, {I(key), I(pv)})) == 0);
    CHECK((load32(pv) & 0xFFFF) == 31 && !fx.wide(load64(pv + 8)).empty());
    CHECK(pg6_win32_prop_variant_clear(com::guest_to_host(pv, 24)) == 0);

    for (const uint64_t o : {store, dev, item, capture, coll}) {
        fx.vcall(o, 2);
    }
    CHECK(fx.vcall(en, 2) == 0);  // the last reference
}

// IAudioClient: the shared mix format, event-driven Initialize, Start without an event fails,
// render-client buffers fill the padding and each period drains 480 frames and sets the event.
void audio_client_render_cycle(AudioFixture& fx) {
    std::vector<uint32_t> sink_frames;
    ga::set_audio_render_sink(
        [](void* user, uint32_t type, const float*, uint32_t frames, uint32_t channels, uint32_t rate) {
            CHECK(type == 0 && channels == 2 && rate == 48000);
            static_cast<std::vector<uint32_t>*>(user)->push_back(frames);
        },
        &sink_frames);
    const uint64_t dev = fx.default_endpoint(fx.enumerator());
    const uint64_t client = fx.activate(dev, kIidAudioClient);
    const uint64_t out = fx.out_slot();
    const uint64_t u32 = com::guest_alloc(8);

    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == static_cast<HResult>(0x88890001u));  // not initialized
    CHECK(as_hr(fx.vcall(client, 8, {I(out)})) == 0);  // GetMixFormat
    const uint64_t fmt = load64(out);
    CHECK((load32(fmt) & 0xFFFF) == 0xFFFE && (load32(fmt) >> 16) == 2 && load32(fmt + 4) == 48000);
    CHECK(as_hr(fx.vcall(client, 7, {I(0), I(fmt), I(out)})) == 0 && load64(out) == 0);  // IsFormatSupported
    const uint64_t periods = com::guest_alloc(16);
    CHECK(as_hr(fx.vcall(client, 9, {I(periods), I(periods + 8)})) == 0 && load64(periods) == 100000);

    // Initialize(shared, EVENTCALLBACK, 0, 0, mix format, NULL)
    CHECK(as_hr(fx.vcall(client, 3, {I(0), I(0x00040000u), I(0), I(0), I(fmt), I(0)})) == 0);
    CHECK(as_hr(fx.vcall(client, 3, {I(0), I(0x00040000u), I(0), I(0), I(fmt), I(0)})) ==
          static_cast<HResult>(0x88890002u));
    pg6_win32_co_task_mem_free(fmt);
    CHECK(as_hr(fx.vcall(client, 4, {I(u32)})) == 0);
    const uint32_t buffer_frames = load32(u32);
    CHECK(buffer_frames >= 480);
    CHECK(as_hr(fx.vcall(client, 10)) == static_cast<HResult>(0x88890014u));  // no event handle yet
    const win32::Handle event = win32::CreateEventW(nullptr, false, false, nullptr);
    CHECK(as_hr(fx.vcall(client, 13, {I(event)})) == 0);

    CHECK(as_hr(fx.vcall(client, 14, {I(fx.guid(kIidAudioRenderClient)), I(out)})) == 0);
    const uint64_t render = load64(out);
    CHECK(as_hr(fx.vcall(client, 14, {I(fx.guid(kIidAudioClock)), I(out)})) == 0);
    const uint64_t clock = load64(out);

    // Pre-fill the whole buffer, then start: every period plays 480 frames and sets the event.
    CHECK(as_hr(fx.vcall(render, 3, {I(buffer_frames + 1), I(out)})) == static_cast<HResult>(0x88890006u));
    CHECK(as_hr(fx.vcall(render, 3, {I(buffer_frames), I(out)})) == 0 && load64(out) != 0);
    CHECK(as_hr(fx.vcall(render, 4, {I(buffer_frames), I(0)})) == 0);
    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == 0 && load32(u32) == buffer_frames);
    CHECK(as_hr(fx.vcall(client, 10)) == 0);
    CHECK(win32::WaitForSingleObject(event, 0) != 0);
    ga::audio_engine_tick();
    CHECK(win32::WaitForSingleObject(event, 0) == 0);
    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == 0 && load32(u32) == buffer_frames - 480);
    CHECK(as_hr(fx.vcall(render, 3, {I(480), I(out)})) == 0);
    CHECK(as_hr(fx.vcall(render, 4, {I(480), I(0)})) == 0);
    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == 0 && load32(u32) == buffer_frames);
    const uint64_t pos = com::guest_alloc(16);
    CHECK(as_hr(fx.vcall(clock, 4, {I(pos), I(pos + 8)})) == 0 && load64(pos) == 480 * 8);

    // Stop freezes the clock; Reset needs a stopped stream and empties it.
    CHECK(as_hr(fx.vcall(client, 12)) == static_cast<HResult>(0x88890005u));
    CHECK(as_hr(fx.vcall(client, 11)) == 0);
    ga::audio_engine_tick();
    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == 0 && load32(u32) == buffer_frames);
    CHECK(as_hr(fx.vcall(client, 12)) == 0);
    CHECK(as_hr(fx.vcall(client, 6, {I(u32)})) == 0 && load32(u32) == 0);

    CHECK(sink_frames.size() == 2 && sink_frames[0] == buffer_frames && sink_frames[1] == 480);
    CHECK(ga::audio_endpoint_stats().render_frames == buffer_frames + 480);
    for (const uint64_t o : {clock, render, client, dev}) {
        fx.vcall(o, 2);
    }
    win32::CloseHandle(event);
}

// ISpatialAudioClient in the executable's sequence: format enumerator -> supported format ->
// ActivateSpatialAudioStream(VT_BLOB params with an event) -> per period Begin, static objects'
// GetBuffer, End; the event is set each period once the stream runs.
void spatial_audio_stream_cycle(AudioFixture& fx) {
    uint32_t sink_calls = 0;
    ga::set_audio_render_sink(
        [](void* user, uint32_t type, const float* pcm, uint32_t frames, uint32_t channels, uint32_t) {
            CHECK(type == 0x2 && channels == 1 && frames == 480 && pcm[7] == 0.25f);
            ++*static_cast<uint32_t*>(user);
        },
        &sink_calls);
    const uint64_t dev = fx.default_endpoint(fx.enumerator());
    const uint64_t sac = fx.activate(dev, kIidSpatialAudioClient);
    const uint64_t out = fx.out_slot();
    const uint64_t u32 = com::guest_alloc(8);

    CHECK(as_hr(fx.vcall(sac, 4, {I(u32)})) == 0 && load32(u32) != 0);  // native static mask
    CHECK(as_hr(fx.vcall(sac, 6, {I(out)})) == 0);
    const uint64_t formats = load64(out);
    CHECK(as_hr(fx.vcall(formats, 3, {I(u32)})) == 0 && load32(u32) >= 1);
    CHECK(as_hr(fx.vcall(formats, 4, {I(0), I(out)})) == 0);
    const uint64_t fmt = load64(out);
    CHECK(as_hr(fx.vcall(sac, 8, {I(fmt)})) == 0);  // IsAudioObjectFormatSupported
    CHECK(as_hr(fx.vcall(sac, 7, {I(fmt), I(u32)})) == 0 && load32(u32) == 480);

    const win32::Handle event = win32::CreateEventW(nullptr, false, false, nullptr);
    const uint64_t params = com::guest_alloc(0x28);
    CHECK(com::write_guest(params, fmt));
    CHECK(com::write_guest(params + 8, uint32_t{0x6}));  // FrontLeft | FrontRight
    CHECK(com::write_guest(params + 20, uint32_t{6}));   // category
    CHECK(com::write_guest(params + 24, uint64_t{event}));
    const uint64_t pv = com::guest_alloc(24);
    CHECK(com::write_guest(pv, uint16_t{65}) && com::write_guest(pv + 8, uint32_t{0x28}) &&
          com::write_guest(pv + 16, params));
    CHECK(as_hr(fx.vcall(sac, 10, {I(pv), I(fx.guid(kIidSpatialStream)), I(out)})) == 0);
    const uint64_t stream = load64(out);
    CHECK(stream != 0);

    CHECK(as_hr(fx.vcall(stream, 5)) == 0);  // Start
    ga::audio_engine_tick();
    CHECK(win32::WaitForSingleObject(event, 0) == 0);

    const uint64_t frames = com::guest_alloc(8);
    uint64_t object = 0;
    for (int pass = 0; pass < 2; ++pass) {
        CHECK(as_hr(fx.vcall(stream, 8, {I(frames), I(frames + 4)})) == 0);  // Begin
        CHECK(load32(frames) == 0 && load32(frames + 4) == 480);
        if (pass == 0) {
            CHECK(as_hr(fx.vcall(stream, 10, {I(0x8), I(out)})) == com::E_INVALIDARG_HR);  // not in the mask
            CHECK(as_hr(fx.vcall(stream, 10, {I(0x2), I(out)})) == 0);
            object = load64(out);
            CHECK(as_hr(fx.vcall(stream, 10, {I(0x2), I(out)})) == com::E_INVALIDARG_HR);  // already active
        }
        const uint64_t buf = com::guest_alloc(16);
        CHECK(as_hr(fx.vcall(object, 3, {I(buf), I(buf + 8)})) == 0);
        CHECK(load64(buf) != 0 && load32(buf + 8) == 480 * 4);
        CHECK(com::write_guest(load64(buf) + 7 * 4, 0.25f));
        CHECK(as_hr(fx.vcall(stream, 9)) == 0);  // End
        if (pass == 1) {
            fx.vcall(object, 2);
        }
    }
    CHECK(as_hr(fx.vcall(stream, 9)) == static_cast<HResult>(0x88890007u));  // End without Begin
    CHECK(sink_calls == 2 && ga::audio_endpoint_stats().spatial_passes == 2);

    CHECK(as_hr(fx.vcall(stream, 6)) == 0);  // Stop: no more events
    ga::audio_engine_tick();
    CHECK(win32::WaitForSingleObject(event, 0) != 0);
    for (const uint64_t o : {stream, formats, sac, dev}) {
        fx.vcall(o, 2);
    }
    win32::CloseHandle(event);
}

// ISpatialAudioObject::SetVolume scales the object's samples (render sink and stereo downmix),
// rejects values outside [0, 1], and the statistics count the non-zero buffers the title wrote.
void spatial_audio_object_volume(AudioFixture& fx) {
    static float seen = -1.0f;
    ga::set_audio_render_sink(
        [](void*, uint32_t, const float* pcm, uint32_t, uint32_t, uint32_t) { seen = pcm[7]; }, nullptr);
    static float left = -1.0f;
    static float right = -1.0f;
    ga::set_audio_stereo_sink(
        [](void*, const float* stereo, uint32_t frames, uint32_t) {
            CHECK(frames == 480);
            left = stereo[2 * 7];
            right = stereo[2 * 7 + 1];
        },
        nullptr);
    const uint64_t dev = fx.default_endpoint(fx.enumerator());
    const uint64_t sac = fx.activate(dev, kIidSpatialAudioClient);
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.vcall(sac, 6, {I(out)})) == 0);
    const uint64_t formats = load64(out);
    CHECK(as_hr(fx.vcall(formats, 4, {I(0), I(out)})) == 0);
    const uint64_t fmt = load64(out);

    const win32::Handle event = win32::CreateEventW(nullptr, false, false, nullptr);
    const uint64_t params = com::guest_alloc(0x28);
    CHECK(com::write_guest(params, fmt) && com::write_guest(params + 8, uint32_t{0x6}) &&
          com::write_guest(params + 20, uint32_t{6}) && com::write_guest(params + 24, uint64_t{event}));
    const uint64_t pv = com::guest_alloc(24);
    CHECK(com::write_guest(pv, uint16_t{65}) && com::write_guest(pv + 8, uint32_t{0x28}) &&
          com::write_guest(pv + 16, params));
    CHECK(as_hr(fx.vcall(sac, 10, {I(pv), I(fx.guid(kIidSpatialStream)), I(out)})) == 0);
    const uint64_t stream = load64(out);
    CHECK(as_hr(fx.vcall(stream, 5)) == 0);  // Start

    const uint64_t frames = com::guest_alloc(8);
    const uint64_t buf = com::guest_alloc(16);
    uint64_t object = 0;
    // pass: volume set before End, sample written, expected output
    struct Pass {
        float volume;
        float sample;
        float expected;
    };
    const Pass passes[] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.25f, 0.125f}, {0.0f, 0.25f, 0.0f}};
    for (const Pass& p : passes) {
        CHECK(as_hr(fx.vcall(stream, 8, {I(frames), I(frames + 4)})) == 0);  // Begin
        if (object == 0) {
            CHECK(as_hr(fx.vcall(stream, 10, {I(0x2), I(out)})) == 0);  // FrontLeft
            object = load64(out);
        }
        CHECK(as_hr(fx.vcall(object, 3, {I(buf), I(buf + 8)})) == 0);
        std::vector<float> zeros(480, 0.0f);
        CHECK(com::write_guest(load64(buf), zeros.data(), zeros.size() * sizeof(float)));
        CHECK(com::write_guest(load64(buf) + 7 * 4, p.sample));
        CHECK(as_hr(fx.vcall(object, 8, {F(p.volume)})) == 0);  // SetVolume
        CHECK(as_hr(fx.vcall(stream, 9)) == 0);                // End
        CHECK(seen == p.expected);
        ga::audio_engine_tick();
        CHECK(left == p.expected && right == 0.0f);
    }
    CHECK(as_hr(fx.vcall(object, 8, {F(1.5f)})) == com::E_INVALIDARG_HR);
    CHECK(as_hr(fx.vcall(object, 8, {F(-0.1f)})) == com::E_INVALIDARG_HR);
    const ga::AudioEndpointStats st = ga::audio_endpoint_stats();
    CHECK(st.spatial_passes == 3 && st.spatial_object_buffers == 3 && st.spatial_nonzero_buffers == 1);

    ga::set_audio_stereo_sink(nullptr, nullptr);
    for (const uint64_t o : {object, stream, formats, sac, dev}) {
        fx.vcall(o, 2);
    }
    win32::CloseHandle(event);
}

void test_mmdevice() {
    AudioFixture fx;
    mmdevice_enumerator_and_endpoint(fx);
    fx.next_case();
    audio_client_render_cycle(fx);
    fx.next_case();
    spatial_audio_stream_cycle(fx);
    fx.next_case();
    spatial_audio_object_volume(fx);
}

// chad.dll XDsp in the executable's call shapes: connect hands out an opaque handle, activation
// fails (no DSP) and clears both outputs, commands are accepted, disconnect forgets the handle.
void test_xdsp() {
    Fixture fx;
    const uint64_t connect = fx.export_address("chad.dll", "XDspConnectWithMaximumStreamLimit");
    const uint64_t activate = fx.export_address("chad.dll", "XDspActivate");
    const uint64_t deactivate = fx.export_address("chad.dll", "XDspDeactivate");
    const uint64_t disconnect = fx.export_address("chad.dll", "XDspDisconnect");
    const uint64_t submit = fx.export_address("chad.dll", "XDspSubmitCommand");
    const uint64_t memory = com::guest_alloc(0x1000);
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(connect, {I(memory), I(0x1000), I(8), I(0x80), I(1), I(out)})) == 0);
    const uint64_t handle = load64(out);
    CHECK(handle != 0 && ga::xdsp_stats().connections == 1);
    CHECK(as_hr(fx.dcall(connect, {I(0), I(0x1000), I(8), I(0x80), I(1), I(out)})) == com::E_INVALIDARG_HR);
    CHECK(load64(out) == 0);

    const uint64_t out_a = fx.out_slot();
    const uint64_t out_b = fx.out_slot();
    CHECK(com::write_guest(out_a, uint64_t{0x77}) && com::write_guest(out_b, uint64_t{0x77}));
    const uint64_t params = com::guest_alloc(0x20);
    CHECK(as_hr(fx.dcall(activate, {I(handle), I(params), I(out_a), I(out_b)})) == com::E_NOTIMPL_HR);
    CHECK(load64(out_a) == 0 && load64(out_b) == 0);
    CHECK(as_hr(fx.dcall(activate, {I(0x1234), I(params), I(out_a), I(out_b)})) == com::E_INVALIDARG_HR);

    const uint64_t command = com::guest_alloc(0x40);
    CHECK(as_hr(fx.dcall(submit, {I(handle), I(command)})) == 0);
    CHECK(as_hr(fx.dcall(submit, {I(handle), I(0)})) == com::E_INVALIDARG_HR);
    CHECK(as_hr(fx.dcall(submit, {I(0x1234), I(command)})) == com::E_INVALIDARG_HR);
    CHECK(ga::xdsp_stats().commands == 1);

    CHECK(as_hr(fx.dcall(deactivate, {I(0)})) == 0);
    CHECK(as_hr(fx.dcall(disconnect, {I(handle)})) == 0);
    CHECK(ga::xdsp_stats().connections == 0);
    CHECK(as_hr(fx.dcall(disconnect, {I(handle)})) == com::E_INVALIDARG_HR);
}

// xapu.dll in the executable's call shapes: commands 1, 2 and 5 are acknowledged with a result
// {type, voice index} in order; other types get none; the count is returned directly.
void test_xapu() {
    Fixture fx;
    const uint64_t connect = fx.export_address("xapu.dll", "XApuConnect");
    const uint64_t enqueue = fx.export_address("xapu.dll", "XApuEnqueueCommand");
    const uint64_t count = fx.export_address("xapu.dll", "XApuGetResultCount");
    const uint64_t dequeue = fx.export_address("xapu.dll", "XApuDequeueResult");
    const uint64_t disconnect = fx.export_address("xapu.dll", "XApuDisconnect");
    const uint64_t params = com::guest_alloc(0x18);
    const uint64_t p2 = com::guest_alloc(0x10);
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(connect, {I(params), I(p2), I(out)})) == 0);
    const uint64_t handle = load64(out);
    CHECK(handle != 0);
    // The second argument receives the shared voice memory: 128 input slots of 0x280 bytes,
    // then 128 output slots of 0x1E00 bytes, zeroed.
    const uint64_t shared = load64(p2);
    CHECK(shared != 0);
    CHECK(load64(shared) == 0 && load64(shared + 0x14000 + 0x7F * 0x1E00 + 0x1DF8) == 0);
    CHECK(as_hr(fx.dcall(connect, {I(params), I(0), I(fx.out_slot())})) == com::E_INVALIDARG_HR);
    CHECK(static_cast<uint32_t>(fx.dcall(count, {I(handle), I(0)})) == 0);

    auto command = [&](uint32_t type, uint32_t voice) {
        const uint64_t c = com::guest_alloc(0x40);
        const uint32_t head[2] = {type, voice};
        CHECK(com::write_guest(c, head, sizeof(head)));
        CHECK(as_hr(fx.dcall(enqueue, {I(handle), I(c), I(0)})) == 0);
    };
    command(5, 7);
    command(9, 3);  // not acknowledged
    command(2, 7);
    CHECK(static_cast<uint32_t>(fx.dcall(count, {I(handle), I(0)})) == 2);
    const uint64_t result = com::guest_alloc(0x38);
    CHECK(com::write_guest(result + 0x30, uint64_t{~0ull}));
    CHECK(as_hr(fx.dcall(dequeue, {I(handle), I(result)})) == 0);
    CHECK(load32(result) == 5 && load32(result + 4) == 7 && load64(result + 0x30) == 0);
    CHECK(as_hr(fx.dcall(dequeue, {I(handle), I(result)})) == 0);
    CHECK(load32(result) == 2 && load32(result + 4) == 7);
    CHECK(as_hr(fx.dcall(dequeue, {I(handle), I(result)})) < 0);  // empty
    CHECK(ga::xapu_stats().commands == 3 && ga::xapu_stats().pending_results == 0);
    // Per-type census: the unknown type 9 is counted as unhandled and posts nothing.
    const ga::XApuStats census = ga::xapu_stats();
    CHECK(census.type5_commands == 1 && census.type2_commands == 1 && census.type1_commands == 0);
    CHECK(census.unhandled_commands == 1 && census.results_posted == 2 && census.type3_results == 0);
    CHECK(as_hr(fx.dcall(disconnect, {I(handle)})) == 0);
    CHECK(static_cast<uint32_t>(fx.dcall(count, {I(handle), I(0)})) == 0);
}

bool xapu_opus_available() { return playgta6::os::xapu_opus::available(); }

// Synthetic Opus packets: `count` 20 ms frames of a 440 Hz sine at 48 kHz, encoded here (no game
// data). Without libopus the packets are placeholders (the model then writes silence).
std::vector<std::vector<uint8_t>> sine_opus_packets(int channels, int count) {
    std::vector<std::vector<uint8_t>> packets;
#if defined(PLAYGTA6_HAVE_OPUS)
    int error = 0;
    OpusEncoder* enc = opus_encoder_create(48000, channels, OPUS_APPLICATION_AUDIO, &error);
    CHECK(enc != nullptr && error == OPUS_OK);
    std::vector<float> pcm(960 * channels);
    for (int f = 0; f < count; ++f) {
        for (int i = 0; i < 960; ++i) {
            const float v = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(f * 960 + i) / 48000.0f);
            for (int c = 0; c < channels; ++c) {
                pcm[i * channels + c] = v;
            }
        }
        std::vector<uint8_t> packet(0x280);
        const int n = opus_encode_float(enc, pcm.data(), 960, packet.data(), static_cast<opus_int32>(packet.size()));
        CHECK(n > 0);
        packet.resize(n > 0 ? n : 1);
        packets.push_back(std::move(packet));
    }
    opus_encoder_destroy(enc);
#else
    for (int f = 0; f < count; ++f) {
        packets.push_back({0xFC, 0xFF, 0xFE});
    }
#endif
    return packets;
}

// Command 3 in the title's layout (0x144F20AD0): an Opus packet in a voice's input slot decodes to
// float32 in its output slot, and the result {3, voice, ..., +0x34 bytes} is queued in order with
// the acknowledgements; +0x20 drops frames from the start; a bad packet posts 0 bytes.
void test_xapu_decode() {
    Fixture fx;
    const uint64_t connect = fx.export_address("xapu.dll", "XApuConnect");
    const uint64_t enqueue = fx.export_address("xapu.dll", "XApuEnqueueCommand");
    const uint64_t count = fx.export_address("xapu.dll", "XApuGetResultCount");
    const uint64_t dequeue = fx.export_address("xapu.dll", "XApuDequeueResult");
    const uint64_t disconnect = fx.export_address("xapu.dll", "XApuDisconnect");
    const uint64_t params = com::guest_alloc(0x18);
    const uint64_t p2 = com::guest_alloc(0x10);
    const uint64_t out = fx.out_slot();
    CHECK(as_hr(fx.dcall(connect, {I(params), I(p2), I(out)})) == 0);
    const uint64_t handle = load64(out);
    const uint64_t shared = load64(p2);
    const ga::XApuStats before = ga::xapu_stats();

    const uint64_t c = com::guest_alloc(0x40);
    const uint64_t result = com::guest_alloc(0x38);
    auto submit = [&](uint32_t type, uint32_t voice) {
        const uint32_t head[2] = {type, voice};
        CHECK(com::write_guest(c, head, sizeof(head)));
        CHECK(as_hr(fx.dcall(enqueue, {I(handle), I(c), I(0)})) == 0);
    };
    // Mirrors 0x144F20AD0: packet copied into input slot `slot`, output in output slot `slot`.
    auto decode = [&](uint32_t voice, uint32_t slot, uint32_t channels, const std::vector<uint8_t>& packet,
                      uint32_t skip) {
        uint8_t cmd[0x40] = {};
        const uint32_t type = 3;
        const float rate = 48000.0f;
        const uint32_t frames = 0x3C0;
        const uint32_t bytes = static_cast<uint32_t>(packet.size());
        const uint64_t input = shared + uint64_t{slot} * 0x280;
        const uint32_t capacity = channels * 0xF00;
        const uint64_t output = shared + 0x14000 + uint64_t{slot} * 0x1E00;
        std::memcpy(cmd + 0x00, &type, 4);
        std::memcpy(cmd + 0x04, &voice, 4);
        std::memcpy(cmd + 0x1C, &rate, 4);
        std::memcpy(cmd + 0x20, &skip, 4);
        std::memcpy(cmd + 0x24, &frames, 4);
        std::memcpy(cmd + 0x28, &bytes, 4);
        std::memcpy(cmd + 0x2C, &input, 8);
        std::memcpy(cmd + 0x34, &capacity, 4);
        std::memcpy(cmd + 0x38, &output, 8);
        CHECK(packet.empty() || com::write_guest(input, packet.data(), packet.size()));
        CHECK(com::write_guest(c, cmd, sizeof(cmd)));
        CHECK(as_hr(fx.dcall(enqueue, {I(handle), I(c), I(0)})) == 0);
    };
    auto next = [&](uint32_t type, uint32_t voice) -> uint32_t {
        CHECK(com::write_guest(result + 0x30, uint64_t{~0ull}));
        CHECK(as_hr(fx.dcall(dequeue, {I(handle), I(result)})) == 0);
        CHECK(load32(result) == type && load32(result + 4) == voice);
        return load32(result + 0x34);
    };
    auto peak = [&](uint32_t slot, uint32_t bytes) {
        std::vector<float> pcm(bytes / 4);
        CHECK(com::read_guest(shared + 0x14000 + uint64_t{slot} * 0x1E00, pcm.data(), bytes));
        float m = 0.0f;
        for (float v : pcm) {
            m = std::max(m, std::fabs(v));
        }
        return m;
    };

    // Mono voice 4 on slot 9: start, five packets (the second trimmed by 312 frames), free.
    const auto mono = sine_opus_packets(1, 5);
    submit(1, 4);
    for (size_t i = 0; i < mono.size(); ++i) {
        decode(4, 9, 1, mono[i], i == 1 ? 312 : 0);
    }
    submit(2, 4);
    CHECK(static_cast<uint32_t>(fx.dcall(count, {I(handle), I(0)})) == 7);
    CHECK(next(1, 4) == 0);
    CHECK(next(3, 4) == 960 * 4);
    CHECK(next(3, 4) == (960 - 312) * 4);
    CHECK(next(3, 4) == 960 * 4);
    CHECK(next(3, 4) == 960 * 4);
    CHECK(next(3, 4) == 960 * 4);
    CHECK(next(2, 4) == 0);
    // The last packet's output is in the slot: the sine once the encoder delay has passed.
    if (xapu_opus_available()) {
        const float m = peak(9, 960 * 4);
        CHECK(m > 0.2f && m < 1.0f);
    }

    // Stereo voice 6 on slot 0x7F: the output fills the whole 0x1E00-byte slot.
    const auto stereo = sine_opus_packets(2, 3);
    for (const auto& packet : stereo) {
        decode(6, 0x7F, 2, packet, 0);
    }
    for (size_t i = 0; i < stereo.size(); ++i) {
        CHECK(next(3, 6) == 0x1E00);
    }
    if (xapu_opus_available()) {
        CHECK(peak(0x7F, 0x1E00) > 0.2f);
    }

    // A packet that does not decode (and an empty one) still gets a result, with 0 bytes.
    if (xapu_opus_available()) {
        decode(5, 1, 1, {0xFF}, 0);  // code-3 TOC without a frame count byte
        CHECK(next(3, 5) == 0);
    }
    decode(5, 1, 1, {}, 0);
    CHECK(next(3, 5) == 0);

    const ga::XApuStats after = ga::xapu_stats();
    CHECK(after.decoded_packets - before.decoded_packets == mono.size() + stereo.size());
    CHECK(after.failed_packets - before.failed_packets == (xapu_opus_available() ? 2u : 1u));
    CHECK(after.decoded_bytes - before.decoded_bytes == (4 * 960 * 4 + (960 - 312) * 4) + 3 * 0x1E00);
    CHECK(after.pending_results == 0);
    // Every decode command posts exactly one type-3 result, decoded or not.
    CHECK(after.type3_results - before.type3_results == after.type3_commands - before.type3_commands);
    CHECK(after.type3_results - before.type3_results ==
          (after.decoded_packets - before.decoded_packets) + (after.failed_packets - before.failed_packets));
    CHECK(after.unhandled_commands == before.unhandled_commands);
    CHECK(as_hr(fx.dcall(disconnect, {I(handle)})) == 0);
}

}  // namespace

int main() {
    test_inflate();
    test_factory_identity_and_stubs();
    test_open_file_and_information();
    test_requests_round_trip();
    test_compressed_and_memory_sources();
    test_errors_and_error_record();
    test_cancel_close_and_signal();
    test_backpressure_and_leaks();
    test_concurrent_queues();
    test_game_input_readings();
    test_game_input_devices();
    test_xg();
    test_acp_hal();
    test_xdsp();
    test_xapu();
    test_xapu_decode();
    test_mmdevice();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xbox_scarlett_guest: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xbox_scarlett_guest: all tests passed\n");
    return 0;
}
