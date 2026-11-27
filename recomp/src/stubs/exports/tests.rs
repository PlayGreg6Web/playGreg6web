use std::fs;
use std::process::Command;

use crate::pe::{ImportModule, ImportedSymbol};
use crate::stubs::tests::{
    MINI_HEADER, MINI_INCLUDE, checkout_catalogue, compile, find_cxx, mini_headers, runtime_root,
    temp_dir,
};
use crate::stubs::{Catalogue, ShimHeader, plan_bridge, render_bridge_namespaced, render_exports};

use super::*;

mod fixture;

/// Guest-state thunks next to the mini shims: `Scale` has a thunk (it takes precedence over the
/// shim of the same name key), `WithCallback` is unmarshalable as a shim but served by its thunk.
const GTHUNK_INCLUDE: &str = "guest/cb.h";
const GTHUNK_HEADER: &str = "#pragma once
extern \"C\" {
void pg6_gthunk_scale(void* state);
void pg6_gthunk_with_callback(void* state);
void pg6_gthunk_table_only(void* state);
}
";

/// An audited family (`pg6_crt_*`, tables only) next to the mini shims.
const CRT_INCLUDE: &str = "mini/pg6_test_crt.h";
const CRT_HEADER: &str = "#pragma once
extern \"C\" {
int pg6_crt_good(int x);
int pg6_crt_bad(int x);
int pg6_crt_wait(int x);
int pg6_crt_unlisted(int x);
}
";

/// Export tables in the runtime's spellings: `E(TAG, \"Name\", symbol)` X-macros and
/// `register_exports(\"DLL\", {export_entry(...)})` lists, plus the audit table.
const TABLE_INCLUDE: &str = "mini/pg6_test_exports.cpp";
const TABLE_SOURCE: &str = r#"
#define PG6_MOD_K32 "KERNEL32.dll"
#define PG6_MOD_KB "KERNELBASE.dll"
#define PG6_MOD_XGR "xgameruntime.dll"
#define PG6_MOD_CRT "ucrtbase.dll"
#define PG6_TEST_EXPORTS(E) \
    E(K32, "GetTickCount", pg6_win32_get_tick_count) \
    E(KB, "GetTickCount", pg6_win32_get_tick_count) \
    E(K32, "Sleep", pg6_win32_sleep) \
    E(K32, "MixArgs", pg6_win32_mix_args) \
    E(K32, "Scale", pg6_win32_scale) \
    E(K32, "VirtualAlloc", pg6_win32_virtual_alloc) \
    E(K32, "PrintfLike", pg6_win32_printf_like) \
    E(K32, "GetPair", pg6_win32_get_pair) \
    E(K32, "WithCallback", pg6_win32_with_callback) \
    E(K32, "Ghost", pg6_win32_ghost) \
    E(XGR, "XTaskQueueSubmitCallback", pg6_gdk_xtask_queue_submit_callback) \
    E(XGR, "SingleDecl", pg6_gdk_single_decl) \
    E(XGR, "SneakyGetTick", pg6_win32_get_tick_count) \
    E(CRT, "good", pg6_crt_good) \
    E(CRT, "bad", pg6_crt_bad) \
    E(CRT, "wait", pg6_crt_wait) \
    E(CRT, "unlisted", pg6_crt_unlisted)

#define PG6_SHIM_AUDIT(X) \
    X(pg6_crt_good, ok, "") \
    X(pg6_crt_bad, deny, "scribbles over guest memory") \
    X(pg6_crt_wait, pending, "not looked at yet")

void register_more() {
    playgta6::os::win32::register_exports("KERNEL32.dll", {
        playgta6::os::win32::export_entry("OrdinalSleep", &pg6_win32_sleep, 7u),
    });
    playgta6::os::win32::register_exports("nofamily.dll", {
        playgta6::os::win32::export_entry("Lonely", &pg6_win32_sleep, 3u),
    });
}
// A guest-state thunk registered under a DLL without a shim family.
static const GuestExport kGuest[] = {
    {"Console.dll", "OnlyInTable", 0, &pg6_gthunk_table_only},
};
"#;

fn headers() -> Vec<ShimHeader> {
    let mut headers = mini_headers();
    headers.push(ShimHeader {
        include: GTHUNK_INCLUDE.to_string(),
        text: GTHUNK_HEADER.to_string(),
    });
    headers.push(ShimHeader {
        include: CRT_INCLUDE.to_string(),
        text: CRT_HEADER.to_string(),
    });
    headers.push(ShimHeader {
        include: TABLE_INCLUDE.to_string(),
        text: TABLE_SOURCE.to_string(),
    });
    headers
}

fn catalogue() -> Catalogue {
    Catalogue::scan(&headers())
}

fn row<'a>(plan: &'a ExportPlan, dll: &str, name: &str) -> &'a ExportRow {
    plan.rows
        .iter()
        .find(|r| dll_key(&r.dll) == dll_key(dll) && r.name == name)
        .unwrap_or_else(|| panic!("{dll}!{name} missing from {:?}", plan.rows))
}

fn none_code(row: &ExportRow) -> &str {
    match &row.binding {
        ExportBinding::None { code, .. } => code,
        other => panic!("{} should have no entry, got {other:?}", row.label()),
    }
}

#[test]
fn plan_binds_every_registered_export_under_the_import_rules() {
    let catalogue = catalogue();
    assert!(
        catalogue.diagnostics.is_empty(),
        "{:?}",
        catalogue.diagnostics
    );
    let plan = plan_exports(&catalogue);

    // Bindable: generated thunks, shared by every DLL spelling of one shim.
    for (dll, name, shim) in [
        ("KERNEL32.dll", "GetTickCount", "pg6_win32_get_tick_count"),
        ("kernelbase.dll", "GetTickCount", "pg6_win32_get_tick_count"),
        ("KERNEL32.dll", "Sleep", "pg6_win32_sleep"),
        ("KERNEL32.dll", "MixArgs", "pg6_win32_mix_args"),
        ("KERNEL32.dll", "VirtualAlloc", "pg6_win32_virtual_alloc"),
        ("KERNEL32.dll", "OrdinalSleep", "pg6_win32_sleep"),
        ("xgameruntime.dll", "SingleDecl", "pg6_gdk_single_decl"),
        ("ucrtbase.dll", "good", "pg6_crt_good"),
    ] {
        assert_eq!(
            row(&plan, dll, name).binding,
            ExportBinding::Generated { shim: shim.into() },
            "{dll}!{name}"
        );
    }

    // A guest-state thunk binds first (also over a marshalable shim) and is listed as is.
    for (dll, name, thunk) in [
        ("KERNEL32.dll", "Scale", "pg6_gthunk_scale"),
        ("KERNEL32.dll", "WithCallback", "pg6_gthunk_with_callback"),
        (
            "xgameruntime.dll",
            "XTaskQueueSubmitCallback",
            "pg6_gthunk_xtask_queue_submit_callback",
        ),
        ("Console.dll", "OnlyInTable", "pg6_gthunk_table_only"),
    ] {
        let expected = if name == "XTaskQueueSubmitCallback" {
            // No such thunk declared in this test: the shim itself takes a host callback.
            None
        } else {
            Some(thunk)
        };
        match expected {
            Some(thunk) => assert_eq!(
                row(&plan, dll, name).binding,
                ExportBinding::GuestState {
                    thunk: thunk.into()
                },
                "{dll}!{name}"
            ),
            None => assert_eq!(none_code(row(&plan, dll, name)), "unmarshalable"),
        }
    }

    // Not bindable: reasons say why.
    assert_eq!(
        none_code(row(&plan, "KERNEL32.dll", "PrintfLike")),
        "unmarshalable"
    );
    assert_eq!(
        none_code(row(&plan, "KERNEL32.dll", "GetPair")),
        "unmarshalable"
    );
    assert_eq!(none_code(row(&plan, "ucrtbase.dll", "bad")), "denied");
    assert_eq!(
        none_code(row(&plan, "ucrtbase.dll", "wait")),
        "pending_audit"
    );
    assert_eq!(
        none_code(row(&plan, "ucrtbase.dll", "unlisted")),
        "pending_audit",
        "an audited family never binds a shim without an audit row"
    );
    assert_eq!(none_code(row(&plan, "KERNEL32.dll", "Ghost")), "undeclared");
    assert_eq!(
        none_code(row(&plan, "xgameruntime.dll", "SneakyGetTick")),
        "no_shim",
        "a shim of another family is not bound under this DLL"
    );
    assert_eq!(
        none_code(row(&plan, "nofamily.dll", "Lonely")),
        "unknown_dll"
    );
    match &row(&plan, "ucrtbase.dll", "bad").binding {
        ExportBinding::None { detail, .. } => {
            assert!(detail.contains("scribbles over guest memory"), "{detail}");
        }
        other => panic!("{other:?}"),
    }

    // Ordinals are carried.
    assert_eq!(row(&plan, "KERNEL32.dll", "OrdinalSleep").ordinal, Some(7));
    assert_eq!(row(&plan, "nofamily.dll", "Lonely").ordinal, Some(3));
    assert_eq!(row(&plan, "KERNEL32.dll", "Sleep").ordinal, None);

    // Counts: 11 callable rows (8 generated + 3 guest-state) on 9 distinct thunks.
    let callable: Vec<_> = plan.rows.iter().filter(|r| r.callee().is_some()).collect();
    assert_eq!(callable.len(), plan.callable_count());
    assert_eq!(plan.callable_count(), 8 + 3);
    assert_eq!(
        plan.distinct_callees(),
        9,
        "GetTickCount and Sleep/OrdinalSleep rows share a shim"
    );
    assert_eq!(plan.generated_shims().len(), 6);
    let reasons = plan.none_by_reason();
    assert_eq!(reasons.get("unmarshalable"), Some(&3));
    assert_eq!(reasons.get("denied"), Some(&1));
    assert_eq!(reasons.get("pending_audit"), Some(&2));
    assert_eq!(reasons.get("undeclared"), Some(&1));
    assert_eq!(reasons.get("no_shim"), Some(&1));
    assert_eq!(reasons.get("unknown_dll"), Some(&1));

    // Sorted by (dll key, name).
    let keys: Vec<(String, String)> = plan
        .rows
        .iter()
        .map(|r| (dll_key(&r.dll), r.name.clone()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
}

#[test]
fn plan_matches_what_a_static_import_of_the_same_export_binds() {
    // The catalogue mode must bind exactly like the import bridge: resolve the same (dll, name)
    // as a static import and compare.
    let catalogue = catalogue();
    let plan = plan_exports(&catalogue);
    let resolver = crate::stubs::ShimResolver::new(&catalogue);
    for row in &plan.rows {
        let import = ImportedSymbol::ByName {
            hint: 0,
            name: row.name.clone(),
        };
        let expected = resolver.resolve(&row.dll, &import);
        match (&row.binding, expected) {
            (
                ExportBinding::Generated { shim } | ExportBinding::GuestState { thunk: shim },
                Resolution::Bound { shim: bound },
            ) => assert_eq!(shim, &bound, "{}", row.label()),
            (ExportBinding::None { .. }, Resolution::Unresolved(_)) => {}
            // Rows the import rules bind but the registry cannot serve (undeclared symbol).
            (ExportBinding::None { code, .. }, Resolution::Bound { .. }) => {
                assert!(
                    code == "undeclared" || code == "shim_mismatch",
                    "{}",
                    row.label()
                );
            }
            (binding, other) => panic!("{}: {binding:?} vs {other:?}", row.label()),
        }
    }
}

#[test]
fn output_is_deterministic_and_lists_every_row() {
    let catalogue = catalogue();
    let plan = plan_exports(&catalogue);
    let a = render_exports(&plan, &catalogue);
    let b = render_exports(&plan_exports(&catalogue), &catalogue);
    assert_eq!(a, b);

    // Scanning the sources in another order changes nothing but possibly the spelling of a DLL
    // with several spellings; the thunk table itself keeps its order.
    let mut reversed = headers();
    reversed.reverse();
    let other = Catalogue::scan(&reversed);
    let c = render_exports(&plan_exports(&other), &other);
    assert_eq!(a.manifest.lines().count(), c.manifest.lines().count());
    let rows = |s: &str| s.lines().filter(|l| l.starts_with("    {\"")).count();
    assert_eq!(rows(&a.source), plan.rows.len());
    assert_eq!(rows(&a.source), rows(&c.source));

    assert!(
        a.source
            .contains("const std::size_t pg6_export_thunk_count = 20;")
    );
    assert!(
        a.header
            .contains("extern const Pg6ExportThunk* const pg6_export_thunks;")
    );
    assert!(
        a.manifest
            .contains("\"format\": \"playgta6-export-thunks\"")
    );
    // Thunks are internal and named after the shim: no clash with `pg6_import_<module>_<n>`.
    assert!(
        a.source
            .contains("void pg6_export_win32_get_tick_count(void* state) {")
    );
    assert!(!a.source.contains("pg6_import_"));
    assert_eq!(
        a.source
            .matches("void pg6_export_win32_get_tick_count(void* state)")
            .count(),
        1,
        "one thunk for both DLL spellings of GetTickCount"
    );
    // The guest-state thunk is listed directly, without a wrapper.
    assert!(a.source.contains("&pg6_gthunk_scale, nullptr},"));
    assert!(!a.source.contains("void pg6_export_gthunk_scale"));
    assert!(a.source.contains("#include \"guest/cb.h\""));
    // Denied rows carry code and detail, no thunk.
    assert!(a.source.contains(
        "\"ucrtbase.dll\", \"bad\", 0u, 0u, nullptr, nullptr, \"denied: pg6_crt_bad: denied by the \
         shim audit: scribbles over guest memory\""
    ));
    // Reasons and counts reach the manifest.
    assert!(
        a.manifest.contains("\"unmarshalable\": 3"),
        "{}",
        a.manifest
    );
    assert!(a.manifest.contains("\"kind\": \"gthunk\""));
}

#[test]
fn empty_catalogue_renders_an_empty_table() {
    let catalogue = Catalogue::scan(&[]);
    let plan = plan_exports(&catalogue);
    assert!(plan.rows.is_empty());
    let out = render_exports(&plan, &catalogue);
    assert!(out.source.contains("pg6_export_thunks = nullptr;"));
    assert!(out.source.contains("pg6_export_thunk_count = 0;"));
    assert!(out.manifest.contains("\"exports\": []"));
}

const HARNESS: &str = r#"// Runs generated export thunks next to a module's import bridge in one executable.
#include "pg6_export_thunks.h"
#include "pg6_import_thunks.h"
#include "pg6_guest_seam.h"
#include "mini/pg6_test_shims.h"
#include "mini/pg6_test_crt.h"
#include "guest/cb.h"

#include <cstdio>
#include <cstring>
#include <string_view>

namespace playgta6::os {
void log(std::string_view message) {
    std::fprintf(stderr, "LOG: %.*s\n", static_cast<int>(message.size()), message.data());
}
}  // namespace playgta6::os

extern "C" {
uint64_t __playgta6_guest_base = 0;
}

namespace {
int g_failures = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #cond);   \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

alignas(16) unsigned char g_state[704];
uint64_t g_stack[32];
const char g_text[] = "hi";
int g_marker = 0;
int g_gthunk_calls = 0;
uint32_t g_slept = 0;
uint64_t g_alloc[4];

uint64_t addr(const void* p) { return pg6_seam_guest_addr(p); }
uint64_t f32_bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
uint64_t f64_bits(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }
uint64_t rax() { return pg6_seam_get_gpr(g_state, 0); }

void reset() {
    std::memset(g_state, 0, sizeof g_state);
    std::memset(g_stack, 0, sizeof g_stack);
    g_stack[8] = 0x1400010FFull;
    pg6_seam_set_gpr(g_state, 4, addr(&g_stack[8]));
}

const Pg6ExportThunk* find_export(const char* dll, const char* name) {
    for (std::size_t i = 0; i < pg6_export_thunk_count; ++i) {
        const Pg6ExportThunk& e = pg6_export_thunks[i];
        if (std::strcmp(e.dll, dll) == 0 && std::strcmp(e.name, name) == 0) {
            return &e;
        }
    }
    return nullptr;
}

void call_export(const char* dll, const char* name) {
    const Pg6ExportThunk* e = find_export(dll, name);
    CHECK(e != nullptr);
    if (e == nullptr || e->thunk == nullptr) {
        CHECK(e != nullptr && e->thunk != nullptr);
        return;
    }
    const uint64_t rsp = pg6_seam_get_gpr(g_state, 4);
    e->thunk(g_state);
    CHECK(pg6_seam_get_gpr(g_state, 4) == rsp + 8);  // popped the return address
}
}  // namespace

extern "C" void pg6_gthunk_scale(void* state) {
    ++g_gthunk_calls;
    pg6_seam_set_gpr(state, 0, 0x5CA1Eu);
    pg6_seam_return(state);
}
extern "C" void pg6_gthunk_with_callback(void*) {}
extern "C" void pg6_gthunk_table_only(void*) {}

extern "C" uint64_t pg6_win32_virtual_alloc(uint64_t address, uint64_t size, uint32_t type,
                                            uint32_t protect) {
    g_alloc[0] = address;
    g_alloc[1] = size;
    g_alloc[2] = type;
    g_alloc[3] = protect;
    return 0x7000000u;
}
extern "C" uint32_t pg6_win32_get_tick_count(void) { return 0xFFFFFFF0u; }
extern "C" void pg6_win32_sleep(uint32_t milliseconds) { g_slept = milliseconds; }
extern "C" double pg6_win32_mix_args(int32_t a, double b, const char* c, float d, uint64_t e,
                                     float f, int8_t g, Pg6OpaqueHandle h) {
    CHECK(a == -5);
    CHECK(b == 2.5);
    CHECK(c == g_text);
    CHECK(d == 1.25f);
    CHECK(e == 0x1122334455667788ull);
    CHECK(f == -3.5f);
    CHECK(g == -2);
    CHECK(h == reinterpret_cast<Pg6OpaqueHandle>(&g_marker));
    return 42.75;
}
extern "C" float pg6_win32_scale(float x, bool flag) { return flag ? x * 2.0f : x; }
extern "C" uint32_t pg6_gdk_single_decl(uint8_t x) noexcept { return x; }
extern "C" int pg6_crt_good(int x) { return x + 1; }

int main() {
    CHECK(pg6_export_thunk_count == EXPECTED_COUNT);

    // Sorted by DLL key (lowercase, no .dll), then name.
    for (std::size_t i = 1; i < pg6_export_thunk_count; ++i) {
        const Pg6ExportThunk& a = pg6_export_thunks[i - 1];
        const Pg6ExportThunk& b = pg6_export_thunks[i];
        (void)a;
        CHECK(b.dll != nullptr && b.name != nullptr);
    }

    // Flags, shim names and reasons.
    const Pg6ExportThunk* tick = find_export("KERNEL32.dll", "GetTickCount");
    CHECK(tick != nullptr && tick->flags == kPg6ExportGenerated && tick->reason == nullptr);
    CHECK(tick != nullptr && std::strcmp(tick->shim, "pg6_win32_get_tick_count") == 0);
    const Pg6ExportThunk* tick_kb = find_export("KERNELBASE.dll", "GetTickCount");
    CHECK(tick != nullptr && tick_kb != nullptr && tick->thunk == tick_kb->thunk);  // one thunk
    const Pg6ExportThunk* scale = find_export("KERNEL32.dll", "Scale");
    CHECK(scale != nullptr && scale->flags == kPg6ExportGuestState);
    CHECK(scale != nullptr && scale->thunk == &pg6_gthunk_scale);
    const Pg6ExportThunk* bad = find_export("ucrtbase.dll", "bad");
    CHECK(bad != nullptr && bad->thunk == nullptr && bad->shim == nullptr && bad->flags == 0);
    CHECK(bad != nullptr && bad->reason != nullptr && std::strncmp(bad->reason, "denied: ", 8) == 0);
    const Pg6ExportThunk* ord = find_export("KERNEL32.dll", "OrdinalSleep");
    CHECK(ord != nullptr && ord->ordinal == 7u);

    // 0 args: unsigned result zero-extended into rax.
    reset();
    pg6_seam_set_gpr(g_state, 0, 0x1234u);
    call_export("KERNEL32.dll", "GetTickCount");
    CHECK(rax() == 0xFFFFFFF0u);
    // The static import of the same shim, from the module bridge, behaves the same.
    {
        const Pg6ImportThunk* import = pg6_find_import_thunk_main(0x3000u);
        CHECK(import != nullptr);
        if (import != nullptr) {
            reset();
            import->thunk(g_state);
            CHECK(rax() == 0xFFFFFFF0u);
        }
    }

    // 4 integer args in rcx rdx r8 r9.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x10000u);
    pg6_seam_set_gpr(g_state, 2, 0x2000u);
    pg6_seam_set_gpr(g_state, 8, 0xDEADBEEF00003000ull);
    pg6_seam_set_gpr(g_state, 9, 0xFFFFFFFF00000004ull);
    call_export("KERNEL32.dll", "VirtualAlloc");
    CHECK(g_alloc[0] == 0x10000u && g_alloc[1] == 0x2000u);
    CHECK(g_alloc[2] == 0x3000u && g_alloc[3] == 4u);
    CHECK(rax() == 0x7000000u);

    // Void result leaves rax; by-ordinal export row uses the same shim.
    reset();
    pg6_seam_set_gpr(g_state, 0, 0x77u);
    pg6_seam_set_gpr(g_state, 1, 0xAAAAAAAA00000064ull);
    call_export("KERNEL32.dll", "OrdinalSleep");
    CHECK(g_slept == 100u);
    CHECK(rax() == 0x77u);

    // 8 mixed args: floats by position, stack slots past the shadow space.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x12345678FFFFFFFBull);
    pg6_seam_set_gpr(g_state, 2, 0x9999u);
    pg6_seam_set_xmm_lo(g_state, 1, f64_bits(2.5));
    pg6_seam_set_gpr(g_state, 8, addr(g_text));
    pg6_seam_set_xmm_lo(g_state, 3, 0xCCCCCCCC00000000ull | f32_bits(1.25f));
    g_stack[8 + 5] = 0x1122334455667788ull;
    g_stack[8 + 6] = 0xAAAAAAAA00000000ull | f32_bits(-3.5f);
    g_stack[8 + 7] = 0x55555555555555FEull;
    g_stack[8 + 8] = addr(&g_marker);
    call_export("KERNEL32.dll", "MixArgs");
    CHECK(pg6_seam_get_xmm_lo(g_state, 0) == f64_bits(42.75));

    // uint8_t parameter, audited CRT shim.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x1FFu);
    call_export("xgameruntime.dll", "SingleDecl");
    CHECK(rax() == 0xFFu);
    reset();
    pg6_seam_set_gpr(g_state, 1, 41u);
    call_export("ucrtbase.dll", "good");
    CHECK(rax() == 42u);

    // The guest-state thunk runs as is.
    reset();
    call_export("KERNEL32.dll", "Scale");
    CHECK(g_gthunk_calls == 1 && rax() == 0x5CA1Eu);

    if (g_failures == 0) {
        std::puts("PG6_EXPORT_THUNKS_OK");
    }
    return g_failures == 0 ? 0 : 1;
}
"#;

#[test]
fn export_table_links_next_to_a_module_bridge_and_marshals_tool_gated() {
    let Some(cxx) = find_cxx() else {
        println!("note: no C++ compiler found ($CXX, c++, g++, clang++, cl); skipping");
        return;
    };
    let catalogue = catalogue();
    let plan = plan_exports(&catalogue);
    let dir = temp_dir("export_compile_run");

    // A module bridge importing the same shims the table exports (and one only it knows):
    // GetTickCount and Sleep as static imports; their thunks are `pg6_import_main_<n>`.
    let mut module = crate::pe::Pe64::parse(&crate::pe::test_util::build_synthetic_pe64(false))
        .expect("synthetic PE");
    module.imports.clear();
    module.delay_imports.clear();
    module.imports.push(ImportModule {
        dll_name: "KERNEL32.dll".to_string(),
        iat_rva: 0x3000,
        symbols: ["GetTickCount", "Sleep"]
            .iter()
            .map(|n| ImportedSymbol::ByName {
                hint: 0,
                name: (*n).to_string(),
            })
            .collect(),
    });
    let bridge_plan = plan_bridge(&module, &catalogue);
    assert_eq!(bridge_plan.bound_count(), 2);
    let bridge_dir = dir.join("stubs");
    render_bridge_namespaced(&bridge_plan, &catalogue, "main")
        .write_to(&bridge_dir)
        .expect("write bridge");
    let export_dir = dir.join("exports");
    render_exports(&plan, &catalogue)
        .write_to(&export_dir)
        .expect("write exports");

    for (rel, text) in [
        (MINI_INCLUDE, MINI_HEADER),
        (CRT_INCLUDE, CRT_HEADER),
        (GTHUNK_INCLUDE, GTHUNK_HEADER),
    ] {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    let harness = HARNESS.replace("EXPECTED_COUNT", &format!("{}u", plan.rows.len()));
    let main_cpp = dir.join("harness.cpp");
    fs::write(&main_cpp, harness).unwrap();
    let exe = dir.join(if cfg!(windows) {
        "harness.exe"
    } else {
        "harness"
    });
    compile(
        &cxx,
        &[
            export_dir.join(EXPORT_SOURCE_FILE),
            bridge_dir.join(crate::stubs::SOURCE_FILE),
            main_cpp,
        ],
        &[
            export_dir.clone(),
            bridge_dir.clone(),
            dir.clone(),
            runtime_root(),
        ],
        Some(&exe),
        true,
        &dir,
    );
    let run = Command::new(&exe).output().expect("run harness");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "harness failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("PG6_EXPORT_THUNKS_OK"),
        "{stdout}\n{stderr}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The table for the checkout's real runtime: every export of every shim family, compiled
/// together in one translation unit.
#[test]
fn export_table_for_real_headers_compiles_tool_gated() {
    let Some(cxx) = find_cxx() else {
        println!("note: no C++ compiler found ($CXX, c++, g++, clang++, cl); skipping");
        return;
    };
    let catalogue = checkout_catalogue();
    let plan = plan_exports(&catalogue);
    let dir = temp_dir("export_compile_real");
    render_exports(&plan, &catalogue)
        .write_to(&dir)
        .expect("write exports");
    compile(
        &cxx,
        &[dir.join(EXPORT_SOURCE_FILE)],
        &[dir.clone(), runtime_root()],
        None,
        false,
        &dir,
    );
    println!(
        "export table of the real runtime: {} exports, {} guest-callable, {} distinct thunks",
        plan.rows.len(),
        plan.callable_count(),
        plan.distinct_callees()
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Budget and the two APIs the real boot found missing: the table needs far fewer synthetic
/// addresses than exist, and `SleepConditionVariableCS` / `WakeAllConditionVariable` (plus the
/// other condition variable APIs) are guest-callable under KERNEL32 and the sync API sets.
#[test]
fn real_catalogue_covers_the_boot_apis_and_fits_the_address_budget() {
    let catalogue = checkout_catalogue();
    let plan = plan_exports(&catalogue);
    assert!(plan.rows.len() > 400, "{} rows", plan.rows.len());
    // 2^20 synthetic addresses (index 0 is the return sentinel); leave room for imports, COM
    // bridge stubs, trampolines and the provider exports: the table may use a tenth of it.
    assert!(
        plan.distinct_callees() < (1 << 20) / 10,
        "{} thunks",
        plan.distinct_callees()
    );
    for name in [
        "SleepConditionVariableCS",
        "SleepConditionVariableSRW",
        "WakeAllConditionVariable",
        "WakeConditionVariable",
        "InitializeConditionVariable",
        "GetProcAddress",
        "GetTickCount64",
        "SetLastError",
        "GetLastError",
    ] {
        let r = row(&plan, "KERNEL32.dll", name);
        assert!(r.callee().is_some(), "{name}: {:?}", r.binding);
    }
    assert!(matches!(
        row(&plan, "KERNEL32.dll", "GetProcAddress").binding,
        ExportBinding::GuestState { .. }
    ));
    // Every denied or pending export has a recorded, human-readable reason.
    for r in &plan.rows {
        if let ExportBinding::None { code, detail } = &r.binding {
            assert!(!code.is_empty() && !detail.is_empty(), "{}", r.label());
        }
    }
}
