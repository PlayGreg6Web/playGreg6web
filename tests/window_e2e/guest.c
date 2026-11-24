/* O29 end-to-end guest: reserves the title-shaped CPU window through XMem, maps pool pages at
   slots 0 | 1 (not host-adjacent) and 1 | 2 (adjacent), writes and reads across them in lifted
   code (straddling accesses), and hands window buffers to host shims (memcpy, strlen) through the
   import bridge. Returns 0x600D when every check passes, else the number of the failed check. */
typedef unsigned long long u64;

__declspec(dllimport) void* XMemVirtualAlloc(void* address, u64 size, unsigned type, unsigned flags, unsigned protect);
__declspec(dllimport) int XMemAllocatePhysicalPages(u64 attributes, u64* count, u64* pfns);
__declspec(dllimport) void* XMemMapPhysicalPages(void* address, u64 count, const u64* pfns);
__declspec(dllimport) int XMemMapPhysicalPagesScatter(const u64* vas, u64 count, const u64* pfns);
#pragma function(memcpy, strlen)
__declspec(dllimport) void* __cdecl memcpy(void* dst, const void* src, u64 size);
__declspec(dllimport) u64 __cdecl strlen(const char* s);

static const char kMsg[] = "window buffers through host shims";
#define K64 0x10000ull

int entry(void) {
    unsigned char* cpu = (unsigned char*)XMemVirtualAlloc(0, 0x400000000ull, 0x20402000u, 0x21u, 4u);
    if (cpu == 0) return 1;
    unsigned char* view = (unsigned char*)XMemVirtualAlloc(0, 16 * K64, 0x00002000u, 0x23u, 4u);
    if (view == 0) return 2;
    u64 pfn[16];
    u64 n = 16;
    if (XMemAllocatePhysicalPages(0x380u, &n, pfn) != 1 || n != 16) return 3;
    if (XMemMapPhysicalPages(view, 16, pfn) != view) return 4;
    u64 vas[3];
    u64 pages[3];
    vas[0] = (u64)cpu;
    vas[1] = (u64)cpu + K64;
    vas[2] = (u64)cpu + 2 * K64;
    pages[0] = pfn[3];
    pages[1] = pfn[9];
    pages[2] = pfn[10];
    if (XMemMapPhysicalPagesScatter(vas, 3, pages) != 1) return 5;

    /* Lifted stores and loads across the split slots 0 | 1. */
    volatile unsigned char* w = cpu;
    for (int i = 0; i < 256; ++i) w[K64 - 128 + i] = (unsigned char)(i * 7 + 1);
    unsigned sum = 0;
    for (int i = 0; i < 256; ++i) sum += w[K64 - 128 + i];
    unsigned expect = 0;
    for (int i = 0; i < 256; ++i) expect += (unsigned char)(i * 7 + 1);
    if (sum != expect) return 6;
    *(volatile u64*)(cpu + K64 - 3) = 0x1122334455667788ull; /* straddling store */
    if (*(volatile u64*)(cpu + K64 - 3) != 0x1122334455667788ull) return 7;
    /* The window aliases the pool pages: slot 0 is pool page 3, slot 1 pool page 9. */
    if (view[3 * K64 + K64 - 3] != 0x88 || view[9 * K64 + 4] != 0x11) return 8;

    /* Host shims on window buffers: split (bounce copy) and contiguous (in place). */
    unsigned char* split = cpu + K64 - 10;
    if (memcpy(split, kMsg, sizeof kMsg) != split) return 9;
    unsigned char* run = cpu + 2 * K64 - 20;
    if (memcpy(run, split, sizeof kMsg) != run) return 10;
    for (unsigned i = 0; i < sizeof kMsg; ++i) {
        if (((volatile unsigned char*)split)[i] != (unsigned char)kMsg[i]) return 11;
        if (((volatile unsigned char*)run)[i] != (unsigned char)kMsg[i]) return 12;
    }
    if (strlen((const char*)split) != sizeof kMsg - 1) return 13;
    if (strlen((const char*)run) != sizeof kMsg - 1) return 14;
    return 0x600D;
}
