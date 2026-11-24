#pragma once

// Shim audit for the import bridge (Track 2, Task O17).
//
// The import bridge (recomp/src/stubs.rs) binds an import of a DLL to a runtime shim through a
// generated marshaling thunk: Win64 integer/pointer/float arguments, top-level guest pointers
// translated, the result written back to rax/xmm0. A shim that can be marshaled is not
// necessarily correct when recompiled MSVC x64 code calls it: a host `va_list`, MSVC struct
// layouts, `long`/`wchar_t` widths, host pointers stored into guest memory, host COM objects,
// callbacks (a guest code address called as a host function pointer). Every shim of an audited
// family (`pg6_crt_`, `pg6_mw_`, `pg6_xbox_`, `pg6_ws2_`) therefore needs a row here, and only
// `ok` rows bind; the bridge reads this table from the header (stubs::scan_tables), so "bound"
// always means "audited". A `deny` or `pending` import stays a logging thunk: its first call
// logs `unresolved import DLL!Name` with the note below and traps, so a boot run reports it.
//
//   X(symbol, verdict, "note")
//     ok       audited: safe to call through a marshaling thunk (note: caveats, if any)
//     deny     must not bind (note: why, and what would make it bindable)
//     pending  not audited yet / waiting for a fix (note: what is missing)
//
// Recurring caveats in the notes:
//   identity  the shim stores a host pointer into guest memory (an out-parameter such as
//             strtol's endptr, or a pointer field of a returned struct). The guest reads it as a
//             guest VA, which is the same address only while __playgta6_guest_base == 0 (the
//             browser build and the default native arena, docs/wasm_abi.md "Guest Address
//             Space"). Returned pointers are always fine: the thunk translates them, and the
//             host -> VA -> host round trip is lossless on every arena.
//   gthunk    a guest-state thunk (pg6_gthunk_*, guest/guest_callbacks.h, guest_crt_exit.h,
//             xbox_scarlett_guest.h) implements the import for guest callers and takes
//             precedence in the bridge; the host shim stays for host callers and GetProcAddress
//             of host code only.
//   host COM  the shim returns an object whose vtable holds host function pointers; guest code
//             would call them as guest addresses (UnknownTarget fault).
//
// PG6_IMPORT_GAPS(X): X("DLL", "Name" | "#ordinal" | "*", "reason") records imports that knowingly
// have no shim yet, so a coverage check can tell a recorded gap from an unexpected one
// (tools/check_import_coverage.py, stubs::Unresolved::recorded).
//
// The tables are data for the bridge; tests/test_shim_audit.cpp expands them against the shim
// headers so every symbol named here exists.

// clang-format off
#define PG6_SHIM_AUDIT(X) \
    /* ---- api-ms-win-crt-convert (O13) ---- */ \
    X(pg6_crt_itoa, ok, "radix 2..36, sign only for radix 10 (MSVC)") \
    X(pg6_crt_strtoi64, ok, "identity: endptr") \
    X(pg6_crt_strtoui64, ok, "identity: endptr") \
    X(pg6_crt_atof, ok, "") \
    X(pg6_crt_atoi, ok, "") \
    X(pg6_crt_atoll, ok, "") \
    X(pg6_crt_strtod, ok, "identity: endptr; ERANGE/EINVAL share MSVC numbering") \
    X(pg6_crt_strtof, ok, "identity: endptr; float result in xmm0") \
    X(pg6_crt_strtol, ok, "Win64 long = int32 with ERANGE clamping; identity: endptr") \
    X(pg6_crt_strtoll, ok, "identity: endptr") \
    X(pg6_crt_strtoul, ok, "Win64 unsigned long = uint32; identity: endptr") \
    X(pg6_crt_strtoull, ok, "identity: endptr") \
    /* ---- api-ms-win-crt-filesystem ---- */ \
    X(pg6_crt_lock_file, ok, "FILE* is an opaque host stream handle (from __acrt_iob_func)") \
    X(pg6_crt_unlock_file, ok, "FILE* is an opaque host stream handle") \
    X(pg6_crt_wstat64, ok, "fills the 56-byte MSVC _stat64 layout; host path without the win32 path mapping, game paths fail with ENOENT") \
    X(pg6_crt_remove, deny, "deletes host files by a raw guest path, bypassing the win32 file shims' path mapping") \
    /* ---- api-ms-win-crt-heap ---- */ \
    X(pg6_crt_set_new_mode, ok, "") \
    X(pg6_crt_calloc, ok, "guest heap (O24 guest_crt family)") \
    X(pg6_crt_free, ok, "guest heap (O24 guest_crt family); foreign pointers are ignored") \
    X(pg6_crt_malloc, ok, "guest heap (O24 guest_crt family)") \
    X(pg6_crt_realloc, ok, "guest heap (O24 guest_crt family)") \
    /* ---- api-ms-win-crt-locale ---- */ \
    X(pg6_crt_configthreadlocale, ok, "") \
    X(pg6_crt_localeconv, ok, "returns a static C-locale MSVC lconv; identity: its string fields are host pointers") \
    /* ---- api-ms-win-crt-math ---- */ \
    X(pg6_crt_setusermatherr, deny, "takes a guest callback (host function-pointer parameter); gthunk: pg6_gthunk_setusermatherr (os/crt_guest.h)") \
    X(pg6_crt_dtest, ok, "MSVC FP_* codes (short)") \
    X(pg6_crt_fdsign, ok, "") \
    X(pg6_crt_fdtest, ok, "MSVC FP_* codes (short)") \
    X(pg6_crt_acos, ok, "") \
    X(pg6_crt_acosf, ok, "") \
    X(pg6_crt_asin, ok, "") \
    X(pg6_crt_asinf, ok, "") \
    X(pg6_crt_atan, ok, "") \
    X(pg6_crt_atan2, ok, "") \
    X(pg6_crt_atan2f, ok, "") \
    X(pg6_crt_atanf, ok, "") \
    X(pg6_crt_cbrtf, ok, "") \
    X(pg6_crt_ceilf, ok, "") \
    X(pg6_crt_cos, ok, "") \
    X(pg6_crt_cosf, ok, "") \
    X(pg6_crt_erff, ok, "") \
    X(pg6_crt_exp, ok, "") \
    X(pg6_crt_exp2, ok, "") \
    X(pg6_crt_exp2f, ok, "") \
    X(pg6_crt_expf, ok, "") \
    X(pg6_crt_fmod, ok, "") \
    X(pg6_crt_fmodf, ok, "") \
    X(pg6_crt_log, ok, "") \
    X(pg6_crt_log10f, ok, "") \
    X(pg6_crt_log2, ok, "") \
    X(pg6_crt_log2f, ok, "") \
    X(pg6_crt_logf, ok, "") \
    X(pg6_crt_lroundf, ok, "Win64 long = int32") \
    X(pg6_crt_modf, ok, "") \
    X(pg6_crt_modff, ok, "") \
    X(pg6_crt_nextafterf, ok, "") \
    X(pg6_crt_nexttowardf, ok, "MSVC long double = double") \
    X(pg6_crt_pow, ok, "") \
    X(pg6_crt_powf, ok, "") \
    X(pg6_crt_remainderf, ok, "") \
    X(pg6_crt_sin, ok, "") \
    X(pg6_crt_sinf, ok, "") \
    X(pg6_crt_sinhf, ok, "") \
    X(pg6_crt_tan, ok, "") \
    X(pg6_crt_tanf, ok, "") \
    /* ---- api-ms-win-crt-runtime (startup order: _set_app_type, __p__commode, _configure_narrow_argv, _initialize_narrow_environment, _initterm_e, _initterm, _get_narrow_winmain_command_line, exit) ---- */ \
    X(pg6_crt_p___argc, ok, "returns the address of a host int (round trip)") \
    X(pg6_crt_p___argv, ok, "argv table and strings are guest heap blocks holding guest VAs (O24)") \
    X(pg6_crt_c_exit, deny, "gthunk: pg6_gthunk__c_exit (guest exit funnel, O22)") \
    X(pg6_crt_cexit, deny, "gthunk: pg6_gthunk__cexit (runs the guest onexit table, O22)") \
    X(pg6_crt_configure_narrow_argv, ok, "returns 0") \
    X(pg6_crt_crt_atexit, deny, "takes a guest callback; gthunk: pg6_gthunk_crt_atexit") \
    X(pg6_crt_errno, ok, "returns the address of the thread's errno int (round trip); every value the CRT shims store is in MSVC numbering (host errno translated by pg6_crt_errno_from_host)") \
    X(pg6_crt__exit, deny, "returns to the caller; gthunk: pg6_gthunk__exit (exit funnel)") \
    X(pg6_crt_get_errno, ok, "") \
    X(pg6_crt_get_narrow_winmain_command_line, ok, "guest heap string (O24)") \
    X(pg6_crt_initialize_narrow_environment, ok, "returns 0") \
    X(pg6_crt_initialize_onexit_table, deny, "gthunk: pg6_gthunk_initialize_onexit_table (guest onexit tables, O22)") \
    X(pg6_crt_initterm, deny, "would call guest function pointers as host code; gthunk: pg6_gthunk_initterm") \
    X(pg6_crt_initterm_e, deny, "would call guest function pointers as host code; gthunk: pg6_gthunk_initterm_e") \
    X(pg6_crt_invalid_parameter_noinfo, ok, "no guest handler can be installed (_set_invalid_parameter_handler is denied): returns, so the failing CRT call reports its error") \
    X(pg6_crt_invalid_parameter_noinfo_noreturn, ok, "returns; the call site is a no-return import, so lifted code traps right after (process ends)") \
    X(pg6_crt_register_onexit_function, deny, "takes a guest callback; gthunk: pg6_gthunk_register_onexit_function") \
    X(pg6_crt_register_thread_local_exe_atexit_callback, deny, "takes a guest callback; gthunk: pg6_gthunk_register_thread_local_exe_atexit_callback") \
    X(pg6_crt_seh_filter_exe, ok, "EXCEPTION_CONTINUE_SEARCH") \
    X(pg6_crt_set_app_type, ok, "") \
    X(pg6_crt_set_errno, ok, "") \
    X(pg6_crt_set_invalid_parameter_handler, deny, "takes a guest callback (host function-pointer parameter); gthunk: pg6_gthunk_set_invalid_parameter_handler (os/crt_guest.h)") \
    X(pg6_crt_abort, ok, "returns after the exit hook; the call site is a no-return import, so lifted code traps right after (process ends)") \
    X(pg6_crt_exit, deny, "runs host atexit callbacks and returns; gthunk: pg6_gthunk_exit (guest exit funnel)") \
    X(pg6_crt_terminate, ok, "returns after the exit hook; no-return import, lifted code traps right after") \
    /* ---- api-ms-win-crt-stdio ---- */ \
    X(pg6_crt_acrt_iob_func, ok, "stable host stdin/stdout/stderr handles, only round-tripped by the guest") \
    X(pg6_crt_p__commode, ok, "returns the address of a host int (round trip)") \
    X(pg6_crt_stdio_common_vfprintf, deny, "host va_list: the guest passes a pointer to 8-byte guest argument slots; gthunk: pg6_gthunk_stdio_common_vfprintf (MSVC formatter over guest slots, os/crt_format.h)") \
    X(pg6_crt_stdio_common_vfwprintf, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vfwprintf (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vsnwprintf_s, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vsnwprintf_s (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vsprintf, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vsprintf (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vsprintf_s, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vsprintf_s (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vsscanf, deny, "host va_list of guest out-pointers; scanf over guest argument slots is not implemented") \
    X(pg6_crt_stdio_common_vswprintf, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vswprintf (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vswprintf_s, deny, "host va_list; gthunk: pg6_gthunk_stdio_common_vswprintf_s (os/crt_format.h)") \
    X(pg6_crt_stdio_common_vswscanf, deny, "host va_list of guest out-pointers; scanf over guest argument slots is not implemented") \
    X(pg6_crt_fseeki64, ok, "") \
    X(pg6_crt_get_stream_buffer_pointers, ok, "per-stream empty {base, ptr, cnt} record in guest memory (guest addresses): basic_filebuf sees no buffered characters and transfers through fgetc/fputc/fwrite") \
    X(pg6_crt_set_fmode, ok, "") \
    X(pg6_crt_wfopen, deny, "opens host files by a raw guest path (and can create/truncate them), bypassing the win32 file shims' path mapping") \
    X(pg6_crt_fclose, ok, "host FILE* handle") \
    X(pg6_crt_fflush, ok, "") \
    X(pg6_crt_fgetc, ok, "") \
    X(pg6_crt_fgetpos, ok, "MSVC fpos_t = int64") \
    X(pg6_crt_fopen, deny, "opens host files by a raw guest path (and can create/truncate them), bypassing the win32 file shims' path mapping") \
    X(pg6_crt_fputc, ok, "") \
    X(pg6_crt_fread, ok, "") \
    X(pg6_crt_fseek, ok, "Win64 long offset = int32") \
    X(pg6_crt_fsetpos, ok, "") \
    X(pg6_crt_ftell, ok, "Win64 long = int32") \
    X(pg6_crt_fwrite, ok, "") \
    X(pg6_crt_rewind, ok, "") \
    X(pg6_crt_setvbuf, ok, "") \
    X(pg6_crt_ungetc, ok, "") \
    /* ---- api-ms-win-crt-string ---- */ \
    X(pg6_crt_stricmp, ok, "ASCII case folding (C locale)") \
    X(pg6_crt_strnicmp, ok, "") \
    X(pg6_crt_wcsicoll, ok, "UTF-16") \
    X(pg6_crt_wcsnicmp, ok, "UTF-16") \
    X(pg6_crt_isalnum, ok, "C locale") \
    X(pg6_crt_isalpha, ok, "C locale") \
    X(pg6_crt_isdigit, ok, "C locale") \
    X(pg6_crt_islower, ok, "C locale") \
    X(pg6_crt_isspace, ok, "C locale") \
    X(pg6_crt_isxdigit, ok, "C locale") \
    X(pg6_crt_strcat, ok, "") \
    X(pg6_crt_strcat_s, ok, "") \
    X(pg6_crt_strcmp, ok, "") \
    X(pg6_crt_strcpy, ok, "") \
    X(pg6_crt_strcpy_s, ok, "") \
    X(pg6_crt_strcspn, ok, "") \
    X(pg6_crt_strlen, ok, "") \
    X(pg6_crt_strncat, ok, "") \
    X(pg6_crt_strncmp, ok, "") \
    X(pg6_crt_strncpy, ok, "") \
    X(pg6_crt_strncpy_s, ok, "_TRUNCATE -> STRUNCATE") \
    X(pg6_crt_strpbrk, ok, "") \
    X(pg6_crt_strtok, ok, "per-thread host state") \
    X(pg6_crt_strtok_s, ok, "identity: the context out-pointer") \
    X(pg6_crt_tolower, ok, "C locale") \
    X(pg6_crt_toupper, ok, "C locale") \
    X(pg6_crt_wcscat_s, ok, "UTF-16") \
    X(pg6_crt_wcscmp, ok, "UTF-16") \
    X(pg6_crt_wcscoll, ok, "UTF-16, C locale") \
    X(pg6_crt_wcslen, ok, "UTF-16") \
    X(pg6_crt_wcsncpy, ok, "UTF-16") \
    /* ---- api-ms-win-crt-time ---- */ \
    X(pg6_crt_difftime64, ok, "") \
    X(pg6_crt_ftime64, ok, "16-byte MSVC __timeb64") \
    X(pg6_crt_gmtime64, ok, "36-byte MSVC tm in per-thread host storage (returned pointer, round trip)") \
    X(pg6_crt_gmtime64_s, ok, "36-byte MSVC tm") \
    X(pg6_crt_localtime64_s, ok, "36-byte MSVC tm converted from the host tm") \
    X(pg6_crt_mkgmtime64, ok, "36-byte MSVC tm") \
    X(pg6_crt_mktime64, ok, "36-byte MSVC tm converted to and from the host tm") \
    X(pg6_crt_time64, ok, "") \
    X(pg6_crt_strftime, ok, "C locale; MSVC tm converted to the host tm") \
    /* ---- api-ms-win-crt-utility ---- */ \
    X(pg6_crt_bsearch, deny, "takes a guest comparator; gthunk: pg6_gthunk_bsearch") \
    X(pg6_crt_qsort, deny, "takes a guest comparator; gthunk: pg6_gthunk_qsort") \
    X(pg6_crt_rand, ok, "MSVC LCG, per thread") \
    X(pg6_crt_srand, ok, "") \
    /* ---- VCRUNTIME140 / VCRUNTIME140_1: EH and setjmp belong to the guest VCRUNTIME (R11/R12 policy) ---- */ \
    X(pg6_crt_cxx_throw_exception, deny, "C++ EH must run in the guest VCRUNTIME (R11): the host stub only calls the exit hook") \
    X(pg6_crt_c_specific_handler, deny, "language-specific handler must be guest code (R11): the stub always continues the search") \
    X(pg6_crt_cxx_frame_handler3, deny, "language-specific handler must be guest code (R11)") \
    X(pg6_crt_cxx_frame_handler4, deny, "language-specific handler must be guest code (R11)") \
    X(pg6_crt_rt_dynamic_cast, deny, "returns null for every non-identical type: wrong dynamic_cast semantics; RTTI walking belongs to the guest VCRUNTIME") \
    X(pg6_crt_current_exception, deny, "EH state belongs to the guest VCRUNTIME") \
    X(pg6_crt_current_exception_context, deny, "EH state belongs to the guest VCRUNTIME") \
    X(pg6_crt_intrinsic_setjmp, deny, "jmp_buf belongs to the guest VCRUNTIME (the stub records only the frame)") \
    X(pg6_crt_std_exception_copy, ok, "copies the message into a guest heap block, stores a guest VA") \
    X(pg6_crt_std_exception_destroy, ok, "frees the guest heap copy") \
    X(pg6_crt_std_terminate, ok, "returns after the exit hook; no-return import, lifted code traps right after") \
    X(pg6_crt_std_type_info_compare, ok, "compares decorated names after the leading '.' (MSVC)") \
    X(pg6_crt_purecall, deny, "returns 0 to the pure-virtual call site instead of terminating") \
    X(pg6_crt_set_purecall_handler, deny, "takes a guest callback (host function-pointer parameter)") \
    X(pg6_crt_longjmp, deny, "jmp_buf belongs to the guest VCRUNTIME (the stub only calls the exit hook)") \
    X(pg6_crt_memchr, ok, "") \
    X(pg6_crt_memcmp, ok, "") \
    X(pg6_crt_memcpy, ok, "") \
    X(pg6_crt_memmove, ok, "") \
    X(pg6_crt_memset, ok, "") \
    X(pg6_crt_strchr, ok, "") \
    X(pg6_crt_strrchr, ok, "") \
    X(pg6_crt_strstr, ok, "") \
    X(pg6_crt_wcsrchr, ok, "UTF-16") \
    /* ---- UCRT beyond O13 (os/ucrt_ext.h, O17): msvcp140's locale/time internals and helpers ---- */ \
    X(pg6_crt_lock_locales, ok, "recursive process-wide lock") \
    X(pg6_crt_unlock_locales, ok, "") \
    X(pg6_crt_lc_codepage_func, ok, "C locale: 0 (CP_ACP, MSVC _CLOCALECP)") \
    X(pg6_crt_lc_collate_cp_func, ok, "C locale: 0") \
    X(pg6_crt_lc_locale_name_func, ok, "6 null locale names (C locale), host static read through the returned pointer") \
    X(pg6_crt_mb_cur_max_func, ok, "C locale: 1") \
    X(pg6_crt_pctype_func, ok, "257-entry MSVC ctype table (entry -1 = EOF), host static read through the returned pointer") \
    X(pg6_crt_setlocale, ok, "C locale only: C, the empty name and NULL succeed, other names fail with NULL") \
    X(pg6_crt_atol, ok, "Win64 long = int32, clamped") \
    X(pg6_crt_btowc, ok, "C locale: byte value, EOF -> WEOF") \
    X(pg6_crt_frexp, ok, "") \
    X(pg6_crt_ldexp, ok, "") \
    X(pg6_crt_isupper, ok, "C locale") \
    X(pg6_crt_iswalnum, ok, "ASCII + Latin-1 letters; false above 0xFF") \
    X(pg6_crt_iswdigit, ok, "ASCII digits") \
    X(pg6_crt_iswspace, ok, "0x09-0x0D, 0x20, 0x85, 0xA0") \
    X(pg6_crt_iswxdigit, ok, "") \
    X(pg6_crt_wcscpy_s, ok, "UTF-16") \
    X(pg6_crt_wcsnlen, ok, "UTF-16") \
    X(pg6_crt_wcsdup, ok, "guest heap (O24)") \
    X(pg6_crt_strncnt, ok, "") \
    X(pg6_crt_aligned_malloc, ok, "guest heap (O24); free() and _aligned_free() release any block") \
    X(pg6_crt_aligned_free, ok, "guest heap (O24)") \
    X(pg6_crt_rand_s, ok, "per-thread PRNG seeded from std::random_device") \
    X(pg6_crt_seh_filter_dll, ok, "EXCEPTION_CONTINUE_SEARCH") \
    X(pg6_crt_fputs, ok, "host FILE* handle") \
    X(pg6_crt_fputwc, ok, "C locale text: units above 0xFF fail with EILSEQ") \
    X(pg6_crt_fgetwc, ok, "C locale text: one byte per unit") \
    X(pg6_crt_ungetwc, ok, "") \
    X(pg6_crt_getdays, ok, "guest heap string, freed by the caller") \
    X(pg6_crt_getmonths, ok, "guest heap string, freed by the caller") \
    X(pg6_crt_gettnames, ok, "opaque zeroed guest heap block, freed by the caller; only passed back to _Strftime") \
    X(pg6_crt_w_getdays, ok, "guest heap string, freed by the caller") \
    X(pg6_crt_w_getmonths, ok, "guest heap string, freed by the caller") \
    X(pg6_crt_w_gettnames, ok, "opaque zeroed guest heap block, freed by the caller") \
    X(pg6_crt_strftime_lc, ok, "C locale names; the _Gettnames block is ignored") \
    X(pg6_crt_wcsftime_lc, ok, "C locale names; literal units copied, specifications formatted narrow and widened") \
    X(pg6_crt_ceil, ok, "") \
    X(pg6_crt_floor, ok, "") \
    X(pg6_crt_sqrt, ok, "") \
    X(pg6_crt_sqrtf, ok, "") \
    X(pg6_crt_cosh, ok, "") \
    X(pg6_crt_sinh, ok, "") \
    X(pg6_crt_expm1, ok, "") \
    X(pg6_crt_log1p, ok, "") \
    X(pg6_crt_wcsncpy_s, ok, "UTF-16; _TRUNCATE -> STRUNCATE") \
    X(pg6_crt_wcsncat_s, ok, "UTF-16; _TRUNCATE -> STRUNCATE") \
    X(pg6_crt_wcstoul, ok, "Win64 unsigned long = uint32; ASCII digits; identity: endptr") \
    X(pg6_crt_itow_s, ok, "UTF-16") \
    X(pg6_crt_p___wargv, ok, "wide argv table and string are guest heap blocks holding guest addresses") \
    X(pg6_crt_configure_wide_argv, ok, "returns 0") \
    X(pg6_crt_invoke_watson, ok, "reports like abort and returns; the call site is a no-return import, so lifted code traps right after") \
    /* ---- libHttpClient.GDK (O14, offline) ---- */ \
    X(pg6_mw_hc_initialize, ok, "") \
    X(pg6_mw_hc_cleanup_async, deny, "completes a guest XAsyncBlock through the host callback path; gthunk: pg6_gthunk_hc_cleanup_async") \
    X(pg6_mw_hc_http_call_create, ok, "synthetic handle") \
    X(pg6_mw_hc_http_call_duplicate_handle, ok, "") \
    X(pg6_mw_hc_http_call_close_handle, ok, "") \
    X(pg6_mw_hc_http_call_perform_async, deny, "completes a guest XAsyncBlock through the host callback path; gthunk: pg6_gthunk_hc_http_call_perform_async") \
    X(pg6_mw_hc_http_call_request_set_url, ok, "") \
    X(pg6_mw_hc_http_call_request_get_url, ok, "identity: string out-pointers") \
    X(pg6_mw_hc_http_call_get_request_url, ok, "identity: string out-pointer") \
    X(pg6_mw_hc_http_call_request_set_request_body_bytes, ok, "") \
    X(pg6_mw_hc_http_call_request_get_request_body_bytes, ok, "identity: buffer out-pointer") \
    X(pg6_mw_hc_http_call_request_set_request_body_string, ok, "") \
    X(pg6_mw_hc_http_call_request_set_header, ok, "") \
    X(pg6_mw_hc_http_call_request_get_num_headers, ok, "") \
    X(pg6_mw_hc_http_call_request_get_header_at_index, ok, "identity: string out-pointers") \
    X(pg6_mw_hc_http_call_request_set_retry_allowed, ok, "") \
    X(pg6_mw_hc_http_call_request_get_retry_allowed, ok, "") \
    X(pg6_mw_hc_http_call_request_set_retry_cache_id, ok, "") \
    X(pg6_mw_hc_http_call_request_get_retry_cache_id, ok, "") \
    X(pg6_mw_hc_http_call_request_set_retry_delay, ok, "") \
    X(pg6_mw_hc_http_call_request_get_retry_delay, ok, "") \
    X(pg6_mw_hc_http_call_request_set_timeout, ok, "") \
    X(pg6_mw_hc_http_call_request_get_timeout, ok, "") \
    X(pg6_mw_hc_http_call_request_set_timeout_window, ok, "") \
    X(pg6_mw_hc_http_call_request_get_timeout_window, ok, "") \
    X(pg6_mw_hc_http_call_response_get_header, ok, "identity: string out-pointer") \
    X(pg6_mw_hc_http_call_response_get_header_at_index, ok, "identity: string out-pointers") \
    X(pg6_mw_hc_http_call_response_get_num_headers, ok, "") \
    X(pg6_mw_hc_http_call_response_get_network_error_code, ok, "offline: HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE)") \
    X(pg6_mw_hc_http_call_response_get_platform_network_error_message, ok, "identity: string out-pointer") \
    X(pg6_mw_hc_http_call_response_get_response_body_bytes, ok, "") \
    X(pg6_mw_hc_http_call_response_get_response_body_bytes_size, ok, "") \
    X(pg6_mw_hc_http_call_response_get_response_string, ok, "identity: string out-pointer") \
    X(pg6_mw_hc_http_call_response_get_status_code, ok, "") \
    X(pg6_mw_hc_http_call_set_tracing, ok, "") \
    X(pg6_mw_hc_settings_get_trace_level, ok, "reports HCTraceLevel::Off") \
    X(pg6_mw_hc_trace_impl_message, deny, "C-variadic: a fixed-signature thunk cannot marshal it") \
    X(pg6_mw_hc_trace_set_etw_enabled, ok, "") \
    /* ---- Bink2Scarlett (O14: synthetic 640x360 clip, no decode, never blocks) ---- */ \
    X(pg6_mw_bink_open_with_options, ok, "returns a runtime-owned BINK whose leading fields follow the public layout (round trip pointer, freed by BinkClose)") \
    X(pg6_mw_bink_close, ok, "") \
    X(pg6_mw_bink_do_frame, ok, "") \
    X(pg6_mw_bink_do_frame_async_multi, ok, "") \
    X(pg6_mw_bink_do_frame_async_wait, ok, "completes at once") \
    X(pg6_mw_bink_next_frame, ok, "") \
    X(pg6_mw_bink_wait, ok, "") \
    X(pg6_mw_bink_pause, ok, "") \
    X(pg6_mw_bink_goto, ok, "") \
    X(pg6_mw_bink_get_key_frame, ok, "") \
    X(pg6_mw_bink_should_skip, ok, "") \
    X(pg6_mw_bink_set_will_loop, ok, "") \
    X(pg6_mw_bink_set_volume, ok, "") \
    X(pg6_mw_bink_get_frame_buffers_info, ok, "writes the BINKFRAMEBUFFERS layout of xbox_middleware.h") \
    X(pg6_mw_bink_register_frame_buffers, ok, "stores the guest plane pointers, never dereferences them") \
    X(pg6_mw_bink_start_async_thread, ok, "") \
    X(pg6_mw_bink_request_stop_async_threads_multi, ok, "") \
    X(pg6_mw_bink_wait_stop_async_threads_multi, ok, "") \
    X(pg6_mw_bink_set_memory, ok, "stores the guest allocator addresses as data, never calls them") \
    X(pg6_mw_bink_set_os_file_callbacks, ok, "stores the guest callback address as data, never calls it") \
    X(pg6_mw_bink_set_sound_system, ok, "stores the guest callback address as data, never calls it") \
    X(pg6_mw_bink_get_platform_info, ok, "zeroes the first 8 bytes of the output") \
    X(pg6_mw_bink_free_globals, ok, "") \
    /* ---- AcpHal ---- */ \
    X(pg6_mw_acp_hal_create, deny, "host COM: returns an IUnknown stub with a host vtable") \
    X(pg6_mw_acp_hal_allocate_shape_contexts, ok, "") \
    X(pg6_mw_acp_hal_release_shape_contexts, ok, "") \
    X(pg6_mw_apu_create_heap, ok, "") \
    X(pg6_mw_apu_alloc, ok, "guest heap block (O24, zero-filled); the out-parameter receives its guest address; APU addresses are synthetic") \
    X(pg6_mw_apu_free, ok, "releases the guest heap block") \
    X(pg6_mw_apu_is_virtual_address_valid, ok, "") \
    X(pg6_mw_apu_map_apu_address, ok, "returns the block address (translated by the thunk)") \
    X(pg6_mw_apu_map_virtual_address, ok, "") \
    /* ---- xapu / chad (loopback sessions) ---- */ \
    X(pg6_mw_xapu_connect, ok, "synthetic handle") \
    X(pg6_mw_xapu_disconnect, ok, "") \
    X(pg6_mw_xapu_enqueue_command, ok, "") \
    X(pg6_mw_xapu_dequeue_result, ok, "") \
    X(pg6_mw_xapu_get_result_count, ok, "") \
    X(pg6_mw_xdsp_connect_with_maximum_stream_limit, ok, "synthetic handle") \
    X(pg6_mw_xdsp_disconnect, ok, "") \
    X(pg6_mw_xdsp_activate, ok, "") \
    X(pg6_mw_xdsp_deactivate, ok, "") \
    X(pg6_mw_xdsp_submit_command, ok, "") \
    /* ---- amd_fidelityfx_loader_xbox ---- */ \
    X(pg6_mw_ffx_create_context, ok, "synthetic context handle; ignores the allocation callbacks") \
    X(pg6_mw_ffx_destroy_context, ok, "") \
    X(pg6_mw_ffx_dispatch, ok, "no-op") \
    X(pg6_mw_ffx_query, deny, "reports FFX_OK without writing any query output (callers read uninitialised out-fields)") \
    /* ---- XboxBvhBuilder (C++ members: this in rcx) ---- */ \
    X(pg6_mw_bvh_builder_ctor, ok, "returns this (MSVC constructor convention)") \
    X(pg6_mw_bvh_builder_dtor, ok, "") \
    X(pg6_mw_bvh_builder_build_batched_2x, ok, "no-op") \
    X(pg6_mw_bvh_builder_copy, ok, "no-op") \
    X(pg6_mw_bvh_builder_get_prebuild_info, ok, "zero sizes (24-byte D3D12 prebuild info)") \
    X(pg6_mw_bvh_builder_rehydrate_batched_2x, ok, "") \
    X(pg6_mw_bvh_builder_set_64bit_morton_codes, ok, "") \
    /* ---- MFPlat / MFReadWrite ---- */ \
    X(pg6_mw_mf_startup, ok, "") \
    X(pg6_mw_mf_shutdown, ok, "") \
    X(pg6_mw_mf_get_system_time, ok, "") \
    X(pg6_mw_mf_create_attributes, deny, "host COM: IUnknown stub with a host vtable") \
    X(pg6_mw_mf_create_media_type, deny, "host COM: IUnknown stub with a host vtable") \
    X(pg6_mw_mf_create_source_resolver, deny, "host COM: IUnknown stub with a host vtable") \
    X(pg6_mw_mf_create_source_reader_from_media_source, deny, "host COM: IUnknown stub with a host vtable") \
    /* ---- xmem / XCRDAPI / xg_xs / GameInput (O12) ---- */ \
    X(pg6_xbox_XMemAlloc, ok, "guest heap (O24); a host allocation hook, when installed, must return guest memory") \
    X(pg6_xbox_XMemAllocDefault, ok, "guest heap (O24)") \
    X(pg6_xbox_XMemFree, ok, "guest heap (O24)") \
    X(pg6_xbox_XMemFreeDefault, ok, "guest heap (O24)") \
    X(pg6_xbox_XMemAllocatePhysicalPages, ok, "page handles are integers") \
    X(pg6_xbox_XMemFreePhysicalPages, ok, "") \
    X(pg6_xbox_XMemMapPhysicalPages, ok, "maps into guest memory") \
    X(pg6_xbox_XMemMapPhysicalPagesScatter, ok, "maps into guest memory") \
    X(pg6_xbox_XMemVirtualAlloc, ok, "guest address space (vmem arena)") \
    X(pg6_xbox_XMemTransferMemory, ok, "") \
    X(pg6_xbox_XMemGetWorkingSetStatistics, ok, "") \
    X(pg6_xbox_XMemIsToolsMemAvailable, ok, "") \
    X(pg6_xbox_DStorageGetFactory, deny, "host COM; gthunk: pg6_gthunk_dstorage_get_factory (guest COM bridge, O23)") \
    X(pg6_xbox_XGCreateTextureComputer, deny, "host COM; gthunk: pg6_gthunk_xg_create_texture_computer (O23)") \
    X(pg6_xbox_XGSuggestSwizzleMode, deny, "wrong call form (the import takes desc, 0, &desc.mode); gthunk: pg6_gthunk_xg_suggest_swizzle_mode (O23)") \
    X(pg6_xbox_GameInputCreate, deny, "host COM; gthunk: pg6_gthunk_game_input_create (O23)") \
    /* ---- WS2_32 (ordinals + getaddrinfo/freeaddrinfo; offline, never blocks) ---- */ \
    X(pg6_ws2_accept, ok, "WSAENOTSOCK") \
    X(pg6_ws2_bind, ok, "WSAENOTSOCK") \
    X(pg6_ws2_closesocket, ok, "WSAENOTSOCK") \
    X(pg6_ws2_connect, ok, "WSAENETUNREACH") \
    X(pg6_ws2_getsockname, ok, "WSAENOTSOCK") \
    X(pg6_ws2_getsockopt, ok, "WSAENOTSOCK") \
    X(pg6_ws2_htons, ok, "") \
    X(pg6_ws2_inet_addr, ok, "") \
    X(pg6_ws2_inet_ntoa, deny, "takes struct in_addr by value (Win64 passes it in a register); the bridge does not marshal structs by value") \
    X(pg6_ws2_listen, ok, "WSAENOTSOCK") \
    X(pg6_ws2_recv, ok, "WSAENETUNREACH") \
    X(pg6_ws2_recvfrom, ok, "WSAENETUNREACH") \
    X(pg6_ws2_select, ok, "fails without blocking") \
    X(pg6_ws2_send, ok, "WSAENETUNREACH") \
    X(pg6_ws2_sendto, ok, "WSAENETUNREACH") \
    X(pg6_ws2_setsockopt, ok, "WSAENOTSOCK") \
    X(pg6_ws2_shutdown, ok, "WSAENOTSOCK") \
    X(pg6_ws2_socket, ok, "INVALID_SOCKET, WSAENETUNREACH") \
    X(pg6_ws2_gethostbyname, ok, "returns null") \
    X(pg6_ws2_gethostname, ok, "") \
    X(pg6_ws2_WSAGetLastError, ok, "per-thread") \
    X(pg6_ws2_WSASetLastError, ok, "per-thread") \
    X(pg6_ws2_WSAStartup, ok, "408-byte 64-bit WSADATA") \
    X(pg6_ws2___WSAFDIsSet, ok, "64-bit fd_set") \
    X(pg6_ws2_getaddrinfo, ok, "fails with WSAENETUNREACH, *result = null") \
    X(pg6_ws2_freeaddrinfo, ok, "")

// Imports of the shipped guest DLLs (msvcp140 & co.) that knowingly have no shim yet.
#define PG6_IMPORT_GAPS(X) \
    X("api-ms-win-crt-runtime-l1-1-0.dll", "_endthreadex", "ends the calling guest thread; the runtime has no guest thread exit path yet (ExitThread)") \
    X("api-ms-win-crt-stdio-l1-1-0.dll", "_fsopen", "host file by a raw guest path, like the denied fopen: needs the win32 file shims' path mapping") \
    X("api-ms-win-crt-stdio-l1-1-0.dll", "_wfsopen", "host file by a raw guest path, like the denied _wfopen") \
    X("api-ms-win-crt-filesystem-l1-1-0.dll", "_wchdir", "host working directory: needs the win32 file shims' path mapping") \
    X("api-ms-win-crt-filesystem-l1-1-0.dll", "_wremove", "host file by a raw guest path, like the denied remove") \
    X("api-ms-win-crt-filesystem-l1-1-0.dll", "_wrename", "host file by a raw guest path, like the denied remove") \
    X("api-ms-win-crt-filesystem-l1-1-0.dll", "_wrmdir", "host directory by a raw guest path, like the denied remove") \
    X("api-ms-win-crt-environment-l1-1-0.dll", "_wgetcwd", "host working directory: needs the win32 file shims' path mapping") \
    X("api-ms-win-crt-heap-l1-1-0.dll", "_aligned_offset_malloc", "no offset-aligned allocation in the guest heap (O24) yet")
// clang-format on
