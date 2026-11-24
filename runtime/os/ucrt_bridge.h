#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// UCRT (api-ms-win-crt-*) & VCRUNTIME140 / VCRUNTIME140_1 C-ABI Bridge (Track 2, Task O13).
//
// Exposes prefixed `extern "C"` `pg6_crt_*` wrappers for all 186 C/C++ runtime imports
// documented in `docs/imports_required.md` without colliding with host libc / MSVC UCRT
// symbols, and registers them into `Win32ModuleRegistry` under their exact DLL names via
// `pg6_ucrt_register_all()`.
//
// Portability invariants:
// - Wide-character APIs operate on UTF-16 (`uint16_t*`), independent of host `wchar_t` width
//   (2 bytes on Windows vs. 4 bytes on Linux and Emscripten Wasm64).
// - Win64 `long` is 32-bit (`int32_t` / `uint32_t`), whereas LP64 Linux/Wasm64 `long` is 64-bit.
//   Signatures use fixed-width integer types matching the Win64 UCRT ABI.
// - Struct layouts (`Pg6CrtTm`, `Pg6CrtTimeb64`, `Pg6CrtStat64`, `Pg6CrtOnexitTable`, etc.)
//   match MSVC x64 UCRT layouts across all host platforms.

extern "C" {

// MSVC secure-CRT constants.
inline constexpr int32_t PG6_CRT_STRUNCATE = 80;

// MSVC <errno.h> values. Every errno the guest can observe (through _errno()/_get_errno() or as
// an errno_t result) uses these, whatever the host's numbering (WASI/Emscripten: EINVAL = 28).
inline constexpr int32_t PG6_CRT_EPERM = 1;
inline constexpr int32_t PG6_CRT_ENOENT = 2;
inline constexpr int32_t PG6_CRT_EIO = 5;
inline constexpr int32_t PG6_CRT_EBADF = 9;
inline constexpr int32_t PG6_CRT_EAGAIN = 11;
inline constexpr int32_t PG6_CRT_ENOMEM = 12;
inline constexpr int32_t PG6_CRT_EACCES = 13;
inline constexpr int32_t PG6_CRT_EEXIST = 17;
inline constexpr int32_t PG6_CRT_ENOTDIR = 20;
inline constexpr int32_t PG6_CRT_EISDIR = 21;
inline constexpr int32_t PG6_CRT_EINVAL = 22;
inline constexpr int32_t PG6_CRT_EMFILE = 24;
inline constexpr int32_t PG6_CRT_ENOSPC = 28;
inline constexpr int32_t PG6_CRT_EDOM = 33;
inline constexpr int32_t PG6_CRT_ERANGE = 34;
inline constexpr int32_t PG6_CRT_ENAMETOOLONG = 38;
inline constexpr int32_t PG6_CRT_ENOTEMPTY = 41;
inline constexpr int32_t PG6_CRT_EILSEQ = 42;
inline constexpr size_t PG6_CRT_TRUNCATE = static_cast<size_t>(-1);

// MSVC UCRT `_dtest` / `_fdtest` classification codes (`<corecrt_math.h>`).
inline constexpr int16_t PG6_CRT_FP_DENORM = -2;
inline constexpr int16_t PG6_CRT_FP_FINITE = -1;
inline constexpr int16_t PG6_CRT_FP_ZERO = 0;
inline constexpr int16_t PG6_CRT_FP_INF = 1;
inline constexpr int16_t PG6_CRT_FP_NAN = 2;

// Exit / termination hook kinds passed to `Pg6CrtExitHook`.
inline constexpr int32_t PG6_CRT_EXIT_NORMAL = 0;
inline constexpr int32_t PG6_CRT_EXIT_QUICK = 1;
inline constexpr int32_t PG6_CRT_EXIT_ABORT = 2;
inline constexpr int32_t PG6_CRT_EXIT_TERMINATE = 3;
inline constexpr int32_t PG6_CRT_EXIT_INVALID_PARAM = 4;
inline constexpr int32_t PG6_CRT_EXIT_PURECALL = 5;
inline constexpr int32_t PG6_CRT_EXIT_CXX_THROW = 6;
inline constexpr int32_t PG6_CRT_EXIT_LONGJMP = 7;

typedef void (*Pg6CrtPvfv)(void);
typedef int32_t (*Pg6CrtPifv)(void);
typedef int32_t (*Pg6CrtOnexitFn)(void);
typedef void (*Pg6CrtInvalidParamHandler)(const uint16_t* expression,
                                          const uint16_t* function,
                                          const uint16_t* file,
                                          uint32_t line,
                                          uintptr_t reserved);
typedef void (*Pg6CrtPurecallHandler)(void);
typedef void (*Pg6CrtUserMathErrHandler)(void* exception_info);
typedef void (*Pg6CrtTlsAtexitCallback)(void* dll_handle, uint32_t reason, void* reserved);
typedef int32_t (*Pg6CrtCompareFn)(const void* a, const void* b);
typedef void (*Pg6CrtExitHook)(int32_t code, int32_t kind);

// Matches MSVC `_onexit_table_t`.
struct Pg6CrtOnexitTable {
    Pg6CrtOnexitFn* first;
    Pg6CrtOnexitFn* last;
    Pg6CrtOnexitFn* end;
};

// Matches Win64 `struct tm` (exact 9 int32_t fields, 36 bytes).
struct Pg6CrtTm {
    int32_t tm_sec;
    int32_t tm_min;
    int32_t tm_hour;
    int32_t tm_mday;
    int32_t tm_mon;
    int32_t tm_year;
    int32_t tm_wday;
    int32_t tm_yday;
    int32_t tm_isdst;
};

// Matches Win64 `struct __timeb64` (16 bytes).
struct Pg6CrtTimeb64 {
    int64_t time;
    uint16_t millitm;
    int16_t timezone;
    int16_t dstflag;
    int16_t _pad;
};

// Matches Win64 `struct __stat64` (56 bytes).
struct Pg6CrtStat64 {
    uint32_t st_dev;
    uint16_t st_ino;
    uint16_t st_mode;
    int16_t st_nlink;
    int16_t st_uid;
    int16_t st_gid;
    uint16_t _pad0;
    uint32_t st_rdev;
    uint32_t _pad1;
    int64_t st_size;
    int64_t atime_sec;
    int64_t mtime_sec;
    int64_t ctime_sec;
};

// Matches Win64 UCRT `struct lconv`.
struct Pg6CrtLconv {
    char* decimal_point;
    char* thousands_sep;
    char* grouping;
    char* int_curr_symbol;
    char* currency_symbol;
    char* mon_decimal_point;
    char* mon_thousands_sep;
    char* mon_grouping;
    char* positive_sign;
    char* negative_sign;
    char int_frac_digits;
    char frac_digits;
    char p_cs_precedes;
    char p_sep_by_space;
    char n_cs_precedes;
    char n_sep_by_space;
    char p_sign_posn;
    char n_sign_posn;
    uint16_t* _W_decimal_point;
    uint16_t* _W_thousands_sep;
    uint16_t* _W_int_curr_symbol;
    uint16_t* _W_currency_symbol;
    uint16_t* _W_mon_decimal_point;
    uint16_t* _W_mon_thousands_sep;
    uint16_t* _W_positive_sign;
    uint16_t* _W_negative_sign;
};

// Matches MSVC `struct __std_exception_data`.
struct Pg6CrtStdExceptionData {
    const char* what;
    uint8_t do_free;
};

// Matches MSVC `struct __std_type_info_data`.
struct Pg6CrtStdTypeInfoData {
    const char* undecorated_name;
    char decorated_name[1];
};

// One registered UCRT/VCRUNTIME export entry for introspection and testing.
struct Pg6UcrtExport {
    const char* module;
    const char* name;
    const char* symbol;
    uint64_t address;
};

// ---------------------------------------------------------------------------
// Registration & test/runtime hook helpers
// ---------------------------------------------------------------------------

// Registers all 186 UCRT & VCRUNTIME140/140_1 symbols into `Win32ModuleRegistry`
// under their exact DLL names from `docs/imports_required.md`. Idempotent.
// Returns the exact count registered (186).
uint32_t pg6_ucrt_register_all(void);

// Returns the static table of all 186 UCRT/VCRUNTIME exports.
const Pg6UcrtExport* pg6_ucrt_export_table(size_t* count);

// Installs an optional hook invoked on exit/_exit/abort/terminate/purecall/throw/longjmp.
// Passing nullptr restores the default safe (non-crashing) behavior. Returns previous hook.
Pg6CrtExitHook pg6_crt_set_exit_hook(Pg6CrtExitHook hook);

// Executes all callbacks in a Pg6CrtOnexitTable in LIFO order and frees its internal storage.
int32_t pg6_crt_execute_onexit_table(Pg6CrtOnexitTable* table);

// Resets mutable runtime hook/counter state for isolated unit tests.
void pg6_crt_reset_runtime_state(void);

// Returns the last code passed to exit/_exit (or 0 if none).
int32_t pg6_crt_get_last_exit_code(void);

// The MSVC number of a host errno value (0 stays 0; unknown values map to EINVAL).
int32_t pg6_crt_errno_from_host(int32_t host_errno);

// ---------------------------------------------------------------------------
// 1. api-ms-win-crt-convert-l1-1-0.dll (12 symbols)
// ---------------------------------------------------------------------------

char* pg6_crt_itoa(int32_t value, char* buffer, int32_t radix);
int64_t pg6_crt_strtoi64(const char* str, char** endptr, int32_t base);
uint64_t pg6_crt_strtoui64(const char* str, char** endptr, int32_t base);
double pg6_crt_atof(const char* str);
int32_t pg6_crt_atoi(const char* str);
int64_t pg6_crt_atoll(const char* str);
double pg6_crt_strtod(const char* str, char** endptr);
float pg6_crt_strtof(const char* str, char** endptr);
int32_t pg6_crt_strtol(const char* str, char** endptr, int32_t base);
int64_t pg6_crt_strtoll(const char* str, char** endptr, int32_t base);
uint32_t pg6_crt_strtoul(const char* str, char** endptr, int32_t base);
uint64_t pg6_crt_strtoull(const char* str, char** endptr, int32_t base);

// ---------------------------------------------------------------------------
// 2. api-ms-win-crt-filesystem-l1-1-0.dll (4 symbols)
// ---------------------------------------------------------------------------

void pg6_crt_lock_file(FILE* stream);
void pg6_crt_unlock_file(FILE* stream);
int32_t pg6_crt_wstat64(const uint16_t* path, Pg6CrtStat64* buffer);
int32_t pg6_crt_remove(const char* path);

// ---------------------------------------------------------------------------
// 3. api-ms-win-crt-heap-l1-1-0.dll (5 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_set_new_mode(int32_t new_mode);
void* pg6_crt_calloc(size_t count, size_t size);
void pg6_crt_free(void* ptr);
void* pg6_crt_malloc(size_t size);
void* pg6_crt_realloc(void* ptr, size_t size);

// ---------------------------------------------------------------------------
// 4. api-ms-win-crt-locale-l1-1-0.dll (2 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_configthreadlocale(int32_t flag);
Pg6CrtLconv* pg6_crt_localeconv(void);

// ---------------------------------------------------------------------------
// 5. api-ms-win-crt-math-l1-1-0.dll (41 symbols)
// ---------------------------------------------------------------------------

void pg6_crt_setusermatherr(Pg6CrtUserMathErrHandler handler);
int16_t pg6_crt_dtest(const double* px);
int32_t pg6_crt_fdsign(float x);
int16_t pg6_crt_fdtest(const float* px);
double pg6_crt_acos(double x);
float pg6_crt_acosf(float x);
double pg6_crt_asin(double x);
float pg6_crt_asinf(float x);
double pg6_crt_atan(double x);
double pg6_crt_atan2(double y, double x);
float pg6_crt_atan2f(float y, float x);
float pg6_crt_atanf(float x);
float pg6_crt_cbrtf(float x);
float pg6_crt_ceilf(float x);
double pg6_crt_cos(double x);
float pg6_crt_cosf(float x);
float pg6_crt_erff(float x);
double pg6_crt_exp(double x);
double pg6_crt_exp2(double x);
float pg6_crt_exp2f(float x);
float pg6_crt_expf(float x);
double pg6_crt_fmod(double x, double y);
float pg6_crt_fmodf(float x, float y);
double pg6_crt_log(double x);
float pg6_crt_log10f(float x);
double pg6_crt_log2(double x);
float pg6_crt_log2f(float x);
float pg6_crt_logf(float x);
int32_t pg6_crt_lroundf(float x);
double pg6_crt_modf(double x, double* iptr);
float pg6_crt_modff(float x, float* iptr);
float pg6_crt_nextafterf(float x, float y);
float pg6_crt_nexttowardf(float x, double y);
double pg6_crt_pow(double x, double y);
float pg6_crt_powf(float x, float y);
float pg6_crt_remainderf(float x, float y);
double pg6_crt_sin(double x);
float pg6_crt_sinf(float x);
float pg6_crt_sinhf(float x);
double pg6_crt_tan(double x);
float pg6_crt_tanf(float x);

// ---------------------------------------------------------------------------
// 6. api-ms-win-crt-runtime-l1-1-0.dll (25 symbols)
// ---------------------------------------------------------------------------

int32_t* pg6_crt_p___argc(void);
char*** pg6_crt_p___argv(void);
void pg6_crt_c_exit(void);
void pg6_crt_cexit(void);
int32_t pg6_crt_configure_narrow_argv(int32_t mode);
int32_t pg6_crt_crt_atexit(Pg6CrtPvfv fn);
int32_t* pg6_crt_errno(void);
void pg6_crt__exit(int32_t status);
int32_t pg6_crt_get_errno(int32_t* value);
char* pg6_crt_get_narrow_winmain_command_line(void);
int32_t pg6_crt_initialize_narrow_environment(void);
int32_t pg6_crt_initialize_onexit_table(Pg6CrtOnexitTable* table);
void pg6_crt_initterm(Pg6CrtPvfv* first, Pg6CrtPvfv* last);
int32_t pg6_crt_initterm_e(Pg6CrtPifv* first, Pg6CrtPifv* last);
void pg6_crt_invalid_parameter_noinfo(void);
void pg6_crt_invalid_parameter_noinfo_noreturn(void);
int32_t pg6_crt_register_onexit_function(Pg6CrtOnexitTable* table, Pg6CrtOnexitFn fn);
void pg6_crt_register_thread_local_exe_atexit_callback(Pg6CrtTlsAtexitCallback callback);
int32_t pg6_crt_seh_filter_exe(uint32_t exception_num, void* exception_ptrs);
void pg6_crt_set_app_type(int32_t app_type);
int32_t pg6_crt_set_errno(int32_t value);
Pg6CrtInvalidParamHandler pg6_crt_set_invalid_parameter_handler(Pg6CrtInvalidParamHandler handler);
void pg6_crt_abort(void);
void pg6_crt_exit(int32_t status);
void pg6_crt_terminate(void);

// ---------------------------------------------------------------------------
// 7. api-ms-win-crt-stdio-l1-1-0.dll (29 symbols)
// ---------------------------------------------------------------------------

FILE* pg6_crt_acrt_iob_func(uint32_t index);
int32_t* pg6_crt_p__commode(void);
int32_t pg6_crt_stdio_common_vfprintf(uint64_t options,
                                      FILE* stream,
                                      const char* format,
                                      void* locale,
                                      va_list arglist);
int32_t pg6_crt_stdio_common_vfwprintf(uint64_t options,
                                       FILE* stream,
                                       const uint16_t* format,
                                       void* locale,
                                       va_list arglist);
int32_t pg6_crt_stdio_common_vsnwprintf_s(uint64_t options,
                                          uint16_t* buffer,
                                          size_t buffer_count,
                                          size_t max_count,
                                          const uint16_t* format,
                                          void* locale,
                                          va_list arglist);
int32_t pg6_crt_stdio_common_vsprintf(uint64_t options,
                                      char* buffer,
                                      size_t buffer_count,
                                      const char* format,
                                      void* locale,
                                      va_list arglist);
int32_t pg6_crt_stdio_common_vsprintf_s(uint64_t options,
                                        char* buffer,
                                        size_t buffer_count,
                                        const char* format,
                                        void* locale,
                                        va_list arglist);
int32_t pg6_crt_stdio_common_vsscanf(uint64_t options,
                                     const char* buffer,
                                     size_t buffer_count,
                                     const char* format,
                                     void* locale,
                                     va_list arglist);
int32_t pg6_crt_stdio_common_vswprintf(uint64_t options,
                                       uint16_t* buffer,
                                       size_t buffer_count,
                                       const uint16_t* format,
                                       void* locale,
                                       va_list arglist);
int32_t pg6_crt_stdio_common_vswprintf_s(uint64_t options,
                                         uint16_t* buffer,
                                         size_t buffer_count,
                                         const uint16_t* format,
                                         void* locale,
                                         va_list arglist);
int32_t pg6_crt_stdio_common_vswscanf(uint64_t options,
                                      const uint16_t* buffer,
                                      size_t buffer_count,
                                      const uint16_t* format,
                                      void* locale,
                                      va_list arglist);
int32_t pg6_crt_fseeki64(FILE* stream, int64_t offset, int32_t origin);
int32_t pg6_crt_get_stream_buffer_pointers(FILE* stream,
                                           char*** base,
                                           char*** ptr,
                                           int32_t** cnt);
int32_t pg6_crt_set_fmode(int32_t mode);
FILE* pg6_crt_wfopen(const uint16_t* filename, const uint16_t* mode);
int32_t pg6_crt_fclose(FILE* stream);
int32_t pg6_crt_fflush(FILE* stream);
int32_t pg6_crt_fgetc(FILE* stream);
int32_t pg6_crt_fgetpos(FILE* stream, int64_t* pos);
FILE* pg6_crt_fopen(const char* filename, const char* mode);
int32_t pg6_crt_fputc(int32_t ch, FILE* stream);
size_t pg6_crt_fread(void* buffer, size_t size, size_t count, FILE* stream);
int32_t pg6_crt_fseek(FILE* stream, int32_t offset, int32_t origin);
int32_t pg6_crt_fsetpos(FILE* stream, const int64_t* pos);
int32_t pg6_crt_ftell(FILE* stream);
size_t pg6_crt_fwrite(const void* buffer, size_t size, size_t count, FILE* stream);
void pg6_crt_rewind(FILE* stream);
int32_t pg6_crt_setvbuf(FILE* stream, char* buffer, int32_t mode, size_t size);
int32_t pg6_crt_ungetc(int32_t ch, FILE* stream);

// ---------------------------------------------------------------------------
// 8. api-ms-win-crt-string-l1-1-0.dll (31 symbols)
// ---------------------------------------------------------------------------

int32_t pg6_crt_stricmp(const char* a, const char* b);
int32_t pg6_crt_strnicmp(const char* a, const char* b, size_t max_count);
int32_t pg6_crt_wcsicoll(const uint16_t* a, const uint16_t* b);
int32_t pg6_crt_wcsnicmp(const uint16_t* a, const uint16_t* b, size_t max_count);
int32_t pg6_crt_isalnum(int32_t c);
int32_t pg6_crt_isalpha(int32_t c);
int32_t pg6_crt_isdigit(int32_t c);
int32_t pg6_crt_islower(int32_t c);
int32_t pg6_crt_isspace(int32_t c);
int32_t pg6_crt_isxdigit(int32_t c);
char* pg6_crt_strcat(char* dst, const char* src);
int32_t pg6_crt_strcat_s(char* dst, size_t dst_size, const char* src);
int32_t pg6_crt_strcmp(const char* a, const char* b);
char* pg6_crt_strcpy(char* dst, const char* src);
int32_t pg6_crt_strcpy_s(char* dst, size_t dst_size, const char* src);
size_t pg6_crt_strcspn(const char* str, const char* control);
size_t pg6_crt_strlen(const char* str);
char* pg6_crt_strncat(char* dst, const char* src, size_t max_count);
int32_t pg6_crt_strncmp(const char* a, const char* b, size_t max_count);
char* pg6_crt_strncpy(char* dst, const char* src, size_t count);
int32_t pg6_crt_strncpy_s(char* dst, size_t dst_size, const char* src, size_t count);
char* pg6_crt_strpbrk(const char* str, const char* charset);
char* pg6_crt_strtok(char* str, const char* delim);
char* pg6_crt_strtok_s(char* str, const char* delim, char** context);
int32_t pg6_crt_tolower(int32_t c);
int32_t pg6_crt_toupper(int32_t c);
int32_t pg6_crt_wcscat_s(uint16_t* dst, size_t dst_size, const uint16_t* src);
int32_t pg6_crt_wcscmp(const uint16_t* a, const uint16_t* b);
int32_t pg6_crt_wcscoll(const uint16_t* a, const uint16_t* b);
size_t pg6_crt_wcslen(const uint16_t* str);
uint16_t* pg6_crt_wcsncpy(uint16_t* dst, const uint16_t* src, size_t count);

// ---------------------------------------------------------------------------
// 9. api-ms-win-crt-time-l1-1-0.dll (9 symbols)
// ---------------------------------------------------------------------------

double pg6_crt_difftime64(int64_t time1, int64_t time0);
void pg6_crt_ftime64(Pg6CrtTimeb64* tp);
Pg6CrtTm* pg6_crt_gmtime64(const int64_t* timer);
int32_t pg6_crt_gmtime64_s(Pg6CrtTm* result, const int64_t* timer);
int32_t pg6_crt_localtime64_s(Pg6CrtTm* result, const int64_t* timer);
int64_t pg6_crt_mkgmtime64(Pg6CrtTm* tm_ptr);
int64_t pg6_crt_mktime64(Pg6CrtTm* tm_ptr);
int64_t pg6_crt_time64(int64_t* timer);
size_t pg6_crt_strftime(char* buffer, size_t max_size, const char* format, const Pg6CrtTm* tm_ptr);

// ---------------------------------------------------------------------------
// 10. api-ms-win-crt-utility-l1-1-0.dll (4 symbols)
// ---------------------------------------------------------------------------

void* pg6_crt_bsearch(const void* key,
                      const void* base,
                      size_t num,
                      size_t size,
                      Pg6CrtCompareFn compare);
void pg6_crt_qsort(void* base, size_t num, size_t size, Pg6CrtCompareFn compare);
int32_t pg6_crt_rand(void);
void pg6_crt_srand(uint32_t seed);

// ---------------------------------------------------------------------------
// 11. VCRUNTIME140.dll (23 symbols) & 12. VCRUNTIME140_1.dll (1 symbol)
// ---------------------------------------------------------------------------

void pg6_crt_cxx_throw_exception(void* exception_object, const void* throw_info);
int32_t pg6_crt_c_specific_handler(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context);
int32_t pg6_crt_cxx_frame_handler3(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context);
int32_t pg6_crt_cxx_frame_handler4(void* exception_record,
                                   void* establisher_frame,
                                   void* context_record,
                                   void* dispatcher_context);
void* pg6_crt_rt_dynamic_cast(void* inptr,
                              int32_t vf_delta,
                              void* src_type,
                              void* target_type,
                              int32_t is_reference);
void** pg6_crt_current_exception(void);
void** pg6_crt_current_exception_context(void);
int32_t pg6_crt_intrinsic_setjmp(void* jmp_buf, void* frame);
void pg6_crt_std_exception_copy(const Pg6CrtStdExceptionData* from, Pg6CrtStdExceptionData* to);
void pg6_crt_std_exception_destroy(Pg6CrtStdExceptionData* data);
void pg6_crt_std_terminate(void);
int32_t pg6_crt_std_type_info_compare(const Pg6CrtStdTypeInfoData* lhs,
                                      const Pg6CrtStdTypeInfoData* rhs);
int32_t pg6_crt_purecall(void);
Pg6CrtPurecallHandler pg6_crt_set_purecall_handler(Pg6CrtPurecallHandler handler);
void pg6_crt_longjmp(void* jmp_buf, int32_t value);
void* pg6_crt_memchr(const void* buf, int32_t val, size_t max_count);
int32_t pg6_crt_memcmp(const void* lhs, const void* rhs, size_t size);
void* pg6_crt_memcpy(void* dst, const void* src, size_t size);
void* pg6_crt_memmove(void* dst, const void* src, size_t size);
void* pg6_crt_memset(void* dst, int32_t val, size_t size);
char* pg6_crt_strchr(const char* str, int32_t ch);
char* pg6_crt_strrchr(const char* str, int32_t ch);
char* pg6_crt_strstr(const char* str, const char* substr);
uint16_t* pg6_crt_wcsrchr(const uint16_t* str, uint16_t ch);

}  // extern "C"

// Calls a host API that reports a host errno through an `int32_t*` (os/guest_heap.h guest_crt::*)
// and stores a reported error into the CRT errno in MSVC numbering.
template <typename F>
auto pg6_crt_with_host_errno(F&& call) {
    int32_t host_errno = 0;
    auto result = call(&host_errno);
    if (host_errno != 0) {
        pg6_crt_set_errno(pg6_crt_errno_from_host(host_errno));
    }
    return result;
}
