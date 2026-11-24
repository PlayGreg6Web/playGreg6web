//! `tests/fixtures/export_thunks/` is real generator output that `tests/test_export_thunks.cpp`
//! links: generated marshaling thunks for real shims, without a build-time dependency on cargo.

use std::fs;
use std::path::Path;

use crate::stubs::tests::checkout_catalogue;

use super::super::*;

/// The exports of the fixture. Regenerate with
/// `playgta6-recomp export-thunks --out tests/fixtures/export_thunks` plus one `--only` per entry
/// (the failure message of the test below prints the command) and delete the generated manifest:
/// it records the number of scanned runtime files, so it would go stale whenever one is added.
const FIXTURE_EXPORTS: &[&str] = &[
    "KERNEL32.dll!GetTickCount64",
    "KERNEL32.dll!SetLastError",
    "KERNEL32.dll!GetLastError",
    "KERNEL32.dll!MultiByteToWideChar",
    "KERNEL32.dll!InitializeCriticalSection",
    "KERNEL32.dll!EnterCriticalSection",
    "KERNEL32.dll!LeaveCriticalSection",
    "KERNEL32.dll!SleepConditionVariableCS",
    "KERNEL32.dll!WakeAllConditionVariable",
    "KERNEL32.dll!InitializeConditionVariable",
    "api-ms-win-crt-string-l1-1-0.dll!strlen",
    "api-ms-win-crt-math-l1-1-0.dll!pow",
    "api-ms-win-crt-math-l1-1-0.dll!sinf",
    "api-ms-win-crt-filesystem-l1-1-0.dll!remove",
    "WS2_32.dll!closesocket",
];

#[test]
fn checked_in_test_fixture_matches_the_generator() {
    let catalogue = checkout_catalogue();
    let selectors: Vec<String> = FIXTURE_EXPORTS.iter().map(|s| (*s).to_string()).collect();
    let plan = plan_exports(&catalogue)
        .only(&selectors)
        .expect("fixture exports are registered");
    assert_eq!(plan.rows.len(), FIXTURE_EXPORTS.len());
    let out = render_exports(&plan, &catalogue);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("export_thunks");
    let normalized = |text: &str| text.replace("\r\n", "\n");
    let mut stale = Vec::new();
    for (name, text) in [
        (EXPORT_HEADER_FILE, &out.header),
        (EXPORT_SOURCE_FILE, &out.source),
        (crate::stubs::SEAM_FILE, &out.seam),
    ] {
        match fs::read_to_string(dir.join(name)) {
            Ok(on_disk) if normalized(&on_disk) == normalized(text) => {}
            _ => stale.push(name),
        }
    }
    let only: Vec<String> = FIXTURE_EXPORTS
        .iter()
        .map(|s| format!("--only \"{s}\""))
        .collect();
    assert!(
        stale.is_empty(),
        "tests/fixtures/export_thunks is stale ({}): regenerate with\n  playgta6-recomp \
         export-thunks --out tests/fixtures/export_thunks {}\nand delete the generated \
         pg6_export_manifest.json",
        stale.join(", "),
        only.join(" ")
    );
}

/// O29: the exports of `tests/fixtures/export_thunks_window/`, thunks for code lifted with the
/// translated CPU window (`tests/test_window_marshal.cpp` drives them with guest buffers split
/// across non-adjacent window pages). Regenerate like the fixture above, with
/// `--out tests/fixtures/export_thunks_window --memory-model window`.
const WINDOW_FIXTURE_EXPORTS: &[&str] = &[
    "KERNEL32.dll!ReadFile",
    "KERNEL32.dll!CreateFileW",
    "KERNEL32.dll!CloseHandle",
    "VCRUNTIME140.dll!memcpy",
    "VCRUNTIME140.dll!memset",
    "api-ms-win-crt-string-l1-1-0.dll!strlen",
    "api-ms-win-crt-string-l1-1-0.dll!strncpy",
    "api-ms-win-crt-string-l1-1-0.dll!strcpy",
    "api-ms-win-crt-string-l1-1-0.dll!strcat",
    "xmem.dll!XMemTransferMemory",
];

#[test]
fn checked_in_window_fixture_matches_the_generator() {
    let catalogue = checkout_catalogue();
    let selectors: Vec<String> = WINDOW_FIXTURE_EXPORTS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let plan = plan_exports(&catalogue)
        .only(&selectors)
        .expect("fixture exports are registered");
    assert_eq!(plan.rows.len(), WINDOW_FIXTURE_EXPORTS.len());
    let out = render_exports_for(
        &plan,
        &catalogue,
        &crate::lift::MemoryModel::translated_window(false),
    );
    assert!(out.source.contains("Pg6SeamBuffers pg6_buffers;"));
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("export_thunks_window");
    let normalized = |text: &str| text.replace("\r\n", "\n");
    let stale: Vec<&str> = [
        (EXPORT_HEADER_FILE, &out.header),
        (EXPORT_SOURCE_FILE, &out.source),
        (crate::stubs::SEAM_FILE, &out.seam),
    ]
    .into_iter()
    .filter(|(name, text)| {
        fs::read_to_string(dir.join(name))
            .map(|d| normalized(&d))
            .ok()
            .as_deref()
            != Some(normalized(text).as_str())
    })
    .map(|(name, _)| name)
    .collect();
    let only: Vec<String> = WINDOW_FIXTURE_EXPORTS
        .iter()
        .map(|s| format!("--only \"{s}\""))
        .collect();
    assert!(
        stale.is_empty(),
        "tests/fixtures/export_thunks_window is stale ({}): regenerate with\n  playgta6-recomp \
         export-thunks --out tests/fixtures/export_thunks_window --memory-model window {}\nand \
         delete the generated pg6_export_manifest.json",
        stale.join(", "),
        only.join(" ")
    );
}
