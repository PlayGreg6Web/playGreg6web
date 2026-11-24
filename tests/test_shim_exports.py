"""Keeps the runtime shim export table (runtime/os/shim_exports.cpp) in sync with the shim headers.

Every `extern "C"` function named pg6_win32_* / pg6_gdk_* declared in runtime/os/*.h (and
pg6_d3d12_* in runtime/gpu/d3d12_xs.h, the d3d12_xs.dll exports) must be
registered in the table or listed in EXCLUDED_SHIMS; the table may not name symbols that do not
exist; and each export name must map back to its shim symbol under the R5 import-bridge rule
(recomp/src/stubs.rs `name_key`: ASCII-lowercase, underscores removed) unless listed in
NAME_EXCEPTIONS.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OS_DIR = REPO_ROOT / "runtime" / "os"
TABLE_CPP = OS_DIR / "shim_exports.cpp"
GPU_SHIM_HEADERS = [REPO_ROOT / "runtime" / "gpu" / "d3d12_xs.h"]

SHIM_RE = re.compile(r"\b(pg6_(?:win32|gdk|d3d12)_\w+)\s*\(")
ENTRY_RE = re.compile(r'^\s*E\((\w+),\s*"([^"]+)",\s*(pg6_(?:win32|gdk|d3d12)_\w+)\)', re.MULTILINE)
MODULE_RE = re.compile(r'#define PG6_MOD_(\w+)\s+"([^"]+)"')

# Declared shims that are intentionally NOT exported: symbol -> reason. Currently every shim
# corresponds to a real API export, so this is empty. Add internal helpers here with a reason
# (e.g. "test hook, not a Win32 API"), never to silence a missing registration.
EXCLUDED_SHIMS: dict[str, str] = {}

# Registered shims whose API name legitimately differs from the shim symbol under name_key:
# symbol -> reason. Empty: every export name matches its symbol under the R5 rule.
NAME_EXCEPTIONS: dict[str, str] = {}

# Shim family of this table's entries per module (one of the families shim_families_for_dll in
# recomp/src/stubs/tables.rs routes the module to; WS2_32.dll tries pg6_ws2_ first).
MODULE_PREFIX = {
    "KERNEL32.dll": "pg6_win32_",
    "ADVAPI32.dll": "pg6_win32_",
    "WS2_32.dll": "pg6_win32_",
    "WINHTTP.dll": "pg6_win32_",
    "bcrypt.dll": "pg6_win32_",
    "USER32.dll": "pg6_win32_",
    "combase.dll": "pg6_win32_",
    "xgameruntime.dll": "pg6_gdk_",
    "d3d12_xs.dll": "pg6_d3d12_",
}
XBL_PREFIX = ("microsoft.xbox.services", "pg6_gdk_")


def name_key(name: str) -> str:
    return name.replace("_", "").lower()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def declared_shims() -> dict[str, str]:
    """symbol -> header file name, for every shim declared inside an extern "C" block."""
    found: dict[str, str] = {}
    for header in sorted(OS_DIR.glob("*.h")) + GPU_SHIM_HEADERS:
        text = strip_comments(header.read_text(encoding="utf-8"))
        for block in re.finditer(r'extern\s+"C"\s*\{', text):
            depth, i = 1, block.end()
            while depth and i < len(text):
                depth += (text[i] == "{") - (text[i] == "}")
                i += 1
            for sym in SHIM_RE.findall(text[block.end() : i]):
                found[sym] = header.name
    return found


def table_entries() -> list[tuple[str, str, str]]:
    """(module name, export name, symbol) for every E(...) line of the table."""
    text = TABLE_CPP.read_text(encoding="utf-8")
    modules = dict(MODULE_RE.findall(text))
    entries = []
    for tag, name, sym in ENTRY_RE.findall(text):
        entries.append((modules[tag], name, sym))
    return entries


class TestShimExportsSync(unittest.TestCase):
    def test_scanner_finds_shims(self) -> None:
        shims = declared_shims()
        self.assertGreaterEqual(len(shims), 100)
        self.assertIn("pg6_win32_create_file_w", shims)
        self.assertIn("pg6_gdk_xtask_queue_create", shims)
        self.assertGreaterEqual(len(table_entries()), 100)

    def test_every_declared_shim_is_registered_or_excluded(self) -> None:
        registered = {sym for _, _, sym in table_entries()}
        missing = sorted(set(declared_shims()) - registered - set(EXCLUDED_SHIMS))
        self.assertEqual(
            missing,
            [],
            "shims declared in runtime/os/*.h but absent from runtime/os/shim_exports.cpp "
            "(add an E(...) line, or list them in EXCLUDED_SHIMS with a reason): "
            + ", ".join(missing),
        )

    def test_no_registered_symbol_is_unknown(self) -> None:
        declared = set(declared_shims())
        unknown = sorted({s for _, _, s in table_entries()} - declared)
        self.assertEqual(unknown, [], "table names symbols not declared in runtime/os/*.h")
        stale = sorted(set(EXCLUDED_SHIMS) - declared)
        self.assertEqual(stale, [], "EXCLUDED_SHIMS lists symbols that no longer exist")
        self.assertEqual(sorted(set(EXCLUDED_SHIMS) & {s for _, _, s in table_entries()}), [])

    def test_no_duplicate_registrations(self) -> None:
        entries = table_entries()
        symbols = [s for _, _, s in entries]
        dup_symbols = sorted({s for s in symbols if symbols.count(s) > 1})
        self.assertEqual(dup_symbols, [], "a shim is registered twice")
        exports = [(m, n) for m, n, _ in entries]
        dup_exports = sorted({e for e in exports if exports.count(e) > 1})
        self.assertEqual(dup_exports, [], "an export name is registered twice in a module")

    def test_export_names_match_symbols_under_r5_rule(self) -> None:
        bad = []
        for module, name, sym in table_entries():
            prefix = "pg6_gdk_" if module.lower().startswith(XBL_PREFIX[0]) else MODULE_PREFIX.get(module)
            if prefix is None or not sym.startswith(prefix):
                bad.append(f"{module}!{name} -> {sym}: wrong shim family for module")
                continue
            if sym in NAME_EXCEPTIONS:
                continue
            if name_key(name) != name_key(sym[len(prefix) :]):
                bad.append(f"{module}!{name} does not map to {sym} (R5 name_key)")
        self.assertEqual(bad, [], "\n".join(bad))
        stale = sorted(set(NAME_EXCEPTIONS) - {s for _, _, s in table_entries()})
        self.assertEqual(stale, [], "NAME_EXCEPTIONS lists unregistered symbols")


# The audited shim families (O17): each lives in its own header and registers itself from its own
# export table; the import bridge reads those tables (recomp/src/stubs/tables.rs scan_tables) and
# binds a shim only with an `ok` row in runtime/os/shim_audit.h.
AUDITED_FAMILIES = [
    ("pg6_crt_", "ucrt_bridge"),
    ("pg6_crt_", "ucrt_ext"),
    ("pg6_mw_", "xbox_middleware"),
    ("pg6_xbox_", "xbox_scarlett"),
    ("pg6_ws2_", "winsock_ordinals"),
]
# Declared but deliberately not exported (host-side hooks): symbol -> reason. They never bind.
AUDITED_FAMILY_INTERNALS: dict[str, str] = {
    "pg6_crt_set_exit_hook": "host-side test/runtime hook",
    "pg6_crt_execute_onexit_table": "host helper; guest imports bind pg6_gthunk_execute_onexit_table",
    "pg6_crt_reset_runtime_state": "test hook",
    "pg6_crt_get_last_exit_code": "test hook",
    "pg6_crt_errno_from_host": "host helper: host errno -> MSVC errno",
    "pg6_ws2_WSASetLastError": "declared, not registered by winsock_ordinals.cpp",
}
AUDIT_ROW_RE = re.compile(r'\bX\((pg6_\w+),\s*(ok|deny|pending),\s*"([^"]*)"\)')
EXPORTED_RE = re.compile(r'"([^"]+)",[\s\\]*(?:\w+\(\s*)?&?(pg6_\w+)')


def family_shims(prefix: str, header: str) -> set[str]:
    text = strip_comments((OS_DIR / f"{header}.h").read_text(encoding="utf-8"))
    found: set[str] = set()
    for block in re.finditer(r'extern\s+"C"\s*\{', text):
        depth, i = 1, block.end()
        while depth and i < len(text):
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        found |= set(re.findall(rf"\b({prefix}\w+)\s*\(", text[block.end() : i]))
    return {s for s in found if not s.endswith("_register_all")}


def family_exports(prefix: str, header: str) -> set[str]:
    text = (OS_DIR / f"{header}.cpp").read_text(encoding="utf-8")
    return {sym for _, sym in EXPORTED_RE.findall(text) if sym.startswith(prefix)}


def audit_rows() -> dict[str, tuple[str, str]]:
    text = (OS_DIR / "shim_audit.h").read_text(encoding="utf-8")
    return {sym: (verdict, note) for sym, verdict, note in AUDIT_ROW_RE.findall(text)}


class TestAuditedFamiliesSync(unittest.TestCase):
    def test_every_family_shim_is_exported_or_internal(self) -> None:
        for prefix, header in AUDITED_FAMILIES:
            declared = family_shims(prefix, header)
            exported = family_exports(prefix, header)
            self.assertGreater(len(exported), 10, prefix)
            missing = sorted(declared - exported - set(AUDITED_FAMILY_INTERNALS))
            self.assertEqual(missing, [], f"{header}.h declares shims its table does not export")
            unknown = sorted(exported - declared)
            self.assertEqual(unknown, [], f"{header}.cpp exports undeclared symbols")

    def test_every_exported_shim_has_an_audit_row(self) -> None:
        rows = audit_rows()
        for prefix, header in AUDITED_FAMILIES:
            missing = sorted(family_exports(prefix, header) - set(rows))
            self.assertEqual(missing, [], "no PG6_SHIM_AUDIT row in runtime/os/shim_audit.h")
        declared = set().union(*(family_shims(p, h) for p, h in AUDITED_FAMILIES))
        stale = sorted(set(rows) - declared)
        self.assertEqual(stale, [], "audit rows for symbols that are not declared")
        for sym, (verdict, note) in rows.items():
            if verdict != "ok":
                self.assertTrue(note.strip(), f"{sym}: {verdict} without a reason")
        stale_internal = sorted(set(AUDITED_FAMILY_INTERNALS) - declared)
        self.assertEqual(stale_internal, [], "AUDITED_FAMILY_INTERNALS lists unknown symbols")


if __name__ == "__main__":
    unittest.main()
