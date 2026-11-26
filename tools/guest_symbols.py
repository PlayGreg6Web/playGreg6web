#!/usr/bin/env python3
"""Name guest addresses for fault reports from the PE's own tables and our runtime tables.

Sources (names only, never code bytes; nothing is written anywhere):
  * PE export table  -> nearest preceding export  (`module!Name+0xOFF`)
  * PE import table  -> IAT slots / thunks         (`import DLL!Func`)
  * PE .pdata        -> function start (RUNTIME_FUNCTION begin) for any address in the image
  * the runtime's own `com bridge: unimplemented <iface> {GUID} slot N (Name)` message
  * docs/HARD_STOPS.md rows (owner-decision status of a COM slot)
Tests use synthetic PEs built in memory (tests/test_fault_report.py).
"""

from __future__ import annotations

import bisect
import re
import struct
from pathlib import Path
from typing import Any, Iterable

UNIMPL_RE = re.compile(
    r"com bridge: unimplemented (?P<iface>\S+) (?P<guid>\{?[0-9A-Fa-f-]{36}\}?) slot (?P<slot>\d+)"
    r"(?: \((?P<name>[^)]*)\))?"
)


class PeSymbols:
    def __init__(self, data: bytes, module: str = "image"):
        self.module = module
        self.exports: list[tuple[int, str]] = []  # (rva, name), sorted
        self.imports: dict[int, str] = {}  # IAT slot rva -> "DLL!Func"
        self.funcs: list[tuple[int, int]] = []  # (begin rva, end rva), sorted
        self.base = 0
        self.size = 0
        self._parse(data)

    @classmethod
    def from_path(cls, path: str | Path) -> "PeSymbols":
        p = Path(path)
        return cls(p.read_bytes(), p.name)

    def _parse(self, d: bytes) -> None:
        if d[:2] != b"MZ":
            raise ValueError("not a PE (no MZ)")
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        if d[pe : pe + 4] != b"PE\0\0":
            raise ValueError("not a PE (no PE signature)")
        (nsec,) = struct.unpack_from("<H", d, pe + 6)
        (opt_size,) = struct.unpack_from("<H", d, pe + 20)
        opt = pe + 24
        (magic,) = struct.unpack_from("<H", d, opt)
        if magic == 0x20B:
            (self.base,) = struct.unpack_from("<Q", d, opt + 24)
            dd, ptr = opt + 112, 8
        elif magic == 0x10B:
            (self.base,) = struct.unpack_from("<I", d, opt + 28)
            dd, ptr = opt + 96, 4
        else:
            raise ValueError("unknown optional header magic")
        (self.size,) = struct.unpack_from("<I", d, opt + 56)
        secs = []
        sp = opt + opt_size
        for i in range(nsec):
            vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", d, sp + 40 * i + 8)
            secs.append((va, max(vsize, rawsize), rawptr))

        def off(rva: int) -> int | None:
            for va, sz, raw in secs:
                if va <= rva < va + sz:
                    return raw + rva - va
            return None

        def cstr(rva: int) -> str:
            o = off(rva)
            if o is None:
                return "?"
            return d[o : d.index(b"\0", o)].decode("ascii", "replace")

        def dirent(i: int) -> tuple[int, int]:
            return struct.unpack_from("<II", d, dd + 8 * i)

        rva, sz = dirent(0)  # exports
        o = off(rva) if rva else None
        if o is not None:
            base_ord, nfunc, nname = struct.unpack_from("<III", d, o + 16)
            fa, na, oa = struct.unpack_from("<III", d, o + 28)
            named: dict[int, str] = {}
            for i in range(nname):
                (nrva,) = struct.unpack_from("<I", d, off(na) + 4 * i)
                (idx,) = struct.unpack_from("<H", d, off(oa) + 2 * i)
                named[idx] = cstr(nrva)
            for i in range(nfunc):
                (frva,) = struct.unpack_from("<I", d, off(fa) + 4 * i)
                if frva and not (rva <= frva < rva + sz):  # skip forwarders
                    self.exports.append((frva, named.get(i, f"#{base_ord + i}")))
            self.exports.sort()

        rva, _ = dirent(1)  # imports
        o = off(rva) if rva else None
        while o is not None:
            oft, _, _, name_rva, iat = struct.unpack_from("<IIIII", d, o)
            if not name_rva:
                break
            dll = cstr(name_rva)
            thunks = off(oft or iat)
            i = 0
            while thunks is not None:
                (v,) = struct.unpack_from("<Q" if ptr == 8 else "<I", d, thunks + ptr * i)
                if v == 0:
                    break
                fn = f"#{v & 0xFFFF}" if v >> (ptr * 8 - 1) else cstr((v & 0x7FFFFFFF) + 2)
                self.imports[iat + ptr * i] = f"{dll}!{fn}"
                i += 1
            o += 20

        rva, sz = dirent(3)  # .pdata
        o = off(rva) if rva else None
        if o is not None:
            for i in range(sz // 12):
                b, e, _ = struct.unpack_from("<III", d, o + 12 * i)
                if b:
                    self.funcs.append((b, e))
            self.funcs.sort()

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.base + self.size

    def describe(self, addr: int) -> str:
        """`module fn 0xSTART+0xOFF nearest export Name+0xOFF`, an import thunk name, or '' outside the image."""
        if not self.contains(addr):
            return ""
        rva = addr - self.base
        for slot, name in self.imports.items():
            if slot <= rva < slot + 8:
                return f"{self.module} import thunk {name}"
        parts = [self.module]
        lo = None
        i = bisect.bisect_right(self.funcs, (rva, 1 << 62)) - 1
        if i >= 0 and self.funcs[i][0] <= rva < self.funcs[i][1]:
            lo = self.funcs[i][0]
            parts.append(f"fn 0x{self.base + lo:X}+0x{rva - lo:x}")
        j = bisect.bisect_right(self.exports, (rva, "\U0010ffff")) - 1
        if j >= 0 and (lo is None or self.exports[j][0] >= lo):
            er, en = self.exports[j]
            parts.append(f"export {en}+0x{rva - er:x}")
        return " ".join(parts)


def find_unimplemented(report: Any, extra_lines: Iterable[str] = ()) -> dict[str, Any] | None:
    """The unimplemented COM slot of a fault: the structured `fault_record.unimplemented_slot` the runtime
    writes (interface, guid, slot, name, optional hard_stop), else the newest `com bridge: unimplemented ...`
    message anywhere in the report (or in extra log lines)."""
    rec = report.get("fault_record") if isinstance(report, dict) else None
    st = rec.get("unimplemented_slot") if isinstance(rec, dict) else None
    if isinstance(st, dict) and isinstance(st.get("slot"), int) and st.get("interface"):
        return {"iface": st["interface"], "guid": st.get("guid"), "slot": st["slot"],
                "name": st.get("name"), "hard_stop": st.get("hard_stop")}
    found: list[dict[str, Any]] = []

    def scan(s: str) -> None:
        for m in UNIMPL_RE.finditer(s):
            found.append({**m.groupdict(), "slot": int(m["slot"])})

    def walk(v: Any) -> None:
        if isinstance(v, str):
            scan(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(report)
    for line in extra_lines:
        scan(line)
    return found[-1] if found else None


def slot_hard_stop(md: str | None, slot: dict[str, Any]) -> str | None:
    """The slot's recorded hard_stop (written into the report by tools/run_native_boot.py), else looked up in `md`."""
    if slot.get("hard_stop"):
        return str(slot["hard_stop"])
    return hard_stop_status(md, slot["iface"], slot["slot"], slot.get("name")) if md else None


def hard_stop_status(md: str, iface: str, slot: int, name: str | None) -> str | None:
    """`Function -> Today` of the docs/HARD_STOPS.md row covering this interface slot, or None."""
    short = iface.split(".")[-1].lower()
    sect = ""
    for line in md.splitlines():
        if line.startswith("## "):
            sect = line.lower().replace("`", "")
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.match(r"^(?:(\w+)\s+)?(\d+(?:\s*/\s*\d+)*)\b", cells[0].lower())
        if not m or slot not in {int(x) for x in re.findall(r"\d+", m.group(2))}:
            continue
        if m.group(1):
            ok = m.group(1) == short
        else:  # slot-only row: the section heading names the interface (whole word, so XStore != XStoreExt)
            ok = re.search(rf"(?<![\w]){re.escape(short)}(?![\w])", sect) is not None
        if ok:
            return f"{cells[1]} -> {cells[2]}"
    if name:
        for line in md.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and len(cells) >= 3 and name in cells[1]:
                return f"{cells[1]} -> {cells[2]}"
    return None
