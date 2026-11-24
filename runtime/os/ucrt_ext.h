#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "os/ucrt_bridge.h"

// UCRT exports beyond O13's 186 (Track 2, Task O17).
//
// The C runtime surface the executable's guest modules need on top of os/ucrt_bridge.h: the
// UCRT internals the shipped msvcp140.dll calls (locale accessors, ctype table, time-name
// strings, _Strftime/_Wcsftime), a few wide-character and stdio helpers, and the aligned
// allocator of the guest heap (O24). Same conventions as ucrt_bridge.h: `pg6_crt_` C ABI,
// Win64 widths (`long` = int32_t, `wchar_t`/`wint_t` = uint16_t), "C" locale, registered under
// the exact API-set DLL names (and ucrtbase.dll) by pg6_ucrt_ext_register_all(), audited in
// os/shim_audit.h.
//
// Guest-visible data
// ------------------
// - __pctype_func() returns a pointer into a 257-entry table of MSVC ctype bits (entry -1 is EOF),
//   ___lc_locale_name_func() a 6-entry array of null names ("C" locale), setlocale() the static
//   string "C": host statics the guest only reads through the returned pointer (the import thunk's
//   host -> guest address translation round-trips on every arena).
// - _Getdays/_Getmonths/_W_Getdays/_W_Getmonths return guest-heap strings in the UCRT format
//   ":Sun:Sunday:Mon:Monday:..." that the caller frees with free(); _Gettnames/_W_Gettnames return
//   a zeroed guest-heap block the caller frees. _Strftime/_Wcsftime ignore that block and format
//   with the "C" locale names.
//
// Locale model: the "C" locale only. setlocale() accepts "C", "" and NULL (query) and fails
// (returns NULL) for anything else; the code pages are 0 (CP_ACP, MSVC's _CLOCALECP) and
// MB_CUR_MAX is 1. Wide classification follows ASCII and Latin-1 for code units below 0x100 and
// is false above.

// MSVC <ctype.h> bits.
inline constexpr uint16_t PG6_CRT_CT_UPPER = 0x1;
inline constexpr uint16_t PG6_CRT_CT_LOWER = 0x2;
inline constexpr uint16_t PG6_CRT_CT_DIGIT = 0x4;
inline constexpr uint16_t PG6_CRT_CT_SPACE = 0x8;
inline constexpr uint16_t PG6_CRT_CT_PUNCT = 0x10;
inline constexpr uint16_t PG6_CRT_CT_CONTROL = 0x20;
inline constexpr uint16_t PG6_CRT_CT_BLANK = 0x40;
inline constexpr uint16_t PG6_CRT_CT_HEX = 0x80;
inline constexpr uint16_t PG6_CRT_CT_LEADBYTE = 0x8000;
inline constexpr uint16_t PG6_CRT_CT_ALPHA = 0x0100 | PG6_CRT_CT_UPPER | PG6_CRT_CT_LOWER;

inline constexpr uint16_t PG6_CRT_WEOF = 0xFFFF;

extern "C" {

// Registers the exports below (all API-set DLLs, mirrored into ucrtbase.dll). Idempotent;
// returns the number of table entries.
uint32_t pg6_ucrt_ext_register_all(void);

// api-ms-win-crt-locale
void pg6_crt_lock_locales(void);
void pg6_crt_unlock_locales(void);
uint32_t pg6_crt_lc_codepage_func(void);
uint32_t pg6_crt_lc_collate_cp_func(void);
uint16_t** pg6_crt_lc_locale_name_func(void);
int32_t pg6_crt_mb_cur_max_func(void);
const uint16_t* pg6_crt_pctype_func(void);
char* pg6_crt_setlocale(int32_t category, const char* locale);

// api-ms-win-crt-convert
int32_t pg6_crt_atol(const char* str);
uint16_t pg6_crt_btowc(int32_t c);

// api-ms-win-crt-math
double pg6_crt_frexp(double x, int32_t* exponent);
double pg6_crt_ldexp(double x, int32_t exponent);

// api-ms-win-crt-string
int32_t pg6_crt_isupper(int32_t c);
int32_t pg6_crt_iswalnum(uint16_t c);
int32_t pg6_crt_iswdigit(uint16_t c);
int32_t pg6_crt_iswspace(uint16_t c);
int32_t pg6_crt_iswxdigit(uint16_t c);
int32_t pg6_crt_wcscpy_s(uint16_t* dst, size_t dst_size, const uint16_t* src);
size_t pg6_crt_wcsnlen(const uint16_t* str, size_t max_count);
uint16_t* pg6_crt_wcsdup(const uint16_t* str);
size_t pg6_crt_strncnt(const char* str, size_t max_count);

// api-ms-win-crt-heap (guest heap, O24)
void* pg6_crt_aligned_malloc(size_t size, size_t alignment);
void pg6_crt_aligned_free(void* ptr);

// api-ms-win-crt-utility
int32_t pg6_crt_rand_s(uint32_t* value);

// api-ms-win-crt-runtime
int32_t pg6_crt_seh_filter_dll(uint32_t exception_num, void* exception_ptrs);

// api-ms-win-crt-stdio (host FILE* handles from __acrt_iob_func)
int32_t pg6_crt_fputs(const char* str, FILE* stream);
uint16_t pg6_crt_fputwc(uint16_t ch, FILE* stream);
uint16_t pg6_crt_fgetwc(FILE* stream);
uint16_t pg6_crt_ungetwc(uint16_t ch, FILE* stream);

// api-ms-win-crt-time (msvcp140's time_get / time_put)
char* pg6_crt_getdays(void);
char* pg6_crt_getmonths(void);
void* pg6_crt_gettnames(void);
uint16_t* pg6_crt_w_getdays(void);
uint16_t* pg6_crt_w_getmonths(void);
void* pg6_crt_w_gettnames(void);
size_t pg6_crt_strftime_lc(char* buffer,
                           size_t max_size,
                           const char* format,
                           const Pg6CrtTm* tm_ptr,
                           void* lc_time);
size_t pg6_crt_wcsftime_lc(uint16_t* buffer,
                           size_t max_size,
                           const uint16_t* format,
                           const Pg6CrtTm* tm_ptr,
                           void* lc_time);

}  // extern "C"
