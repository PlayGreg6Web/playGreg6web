#!/usr/bin/env python3
"""Print the faulting host stack from a playgta6_boot fault minidump.

Usage: minidump_stack.py <dump.dmp | boot-report.json> [--pdb-dir DIR] [--cdb PATH]

A report argument uses fault_record.minidump. Uses cdb.exe when installed (best: full symbols),
otherwise parses the dump's exception stream and stack memory with the stdlib and prints
module+offset frames (resolve with the harness PDB via cdb/WinDbg when available).
Dumps hold no full memory and stay local: do not commit them.
"""
import argparse, json, os, shutil, struct, subprocess, sys

STREAM_THREADS, STREAM_MODULES, STREAM_EXCEPTION, STREAM_MEMORY = 3, 4, 6, 5


def resolve_dump(arg):
    if arg.lower().endswith(".json"):
        with open(arg, encoding="utf-8") as f:
            path = ((json.load(f).get("fault_record") or {}).get("minidump"))
        if not path:
            sys.exit("report has no fault_record.minidump")
        return path
    return arg


def find_cdb(explicit):
    if explicit:
        return explicit
    found = shutil.which("cdb.exe")
    if found:
        return found
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if root:
            p = os.path.join(root, "Windows Kits", "10", "Debuggers", "x64", "cdb.exe")
            if os.path.exists(p):
                return p
    return None


def parse(data):
    sig, _ver, nstreams, dir_rva = struct.unpack_from("<4sIII", data, 0)
    if sig != b"MDMP":
        sys.exit("not a minidump")
    streams = {}
    for i in range(nstreams):
        t, size, rva = struct.unpack_from("<III", data, dir_rva + 12 * i)
        streams[t] = (size, rva)
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--pdb-dir")
    ap.add_argument("--cdb")
    args = ap.parse_args()
    path = resolve_dump(args.dump)
    cdb = find_cdb(args.cdb)
    if cdb:
        sym = ".sympath+ " + args.pdb_dir + ";" if args.pdb_dir else ""
        cmd = [cdb, "-z", path, "-c", sym + ".ecxr;kn 40;q"]
        sys.exit(subprocess.call(cmd))
    data = open(path, "rb").read()
    streams = parse(data)
    modules = []
    if STREAM_MODULES in streams:
        _s, rva = streams[STREAM_MODULES]
        (n,) = struct.unpack_from("<I", data, rva)
        for i in range(n):
            base, size, _cs, _ts, name_rva = struct.unpack_from("<QIIII", data, rva + 4 + 108 * i)
            (nlen,) = struct.unpack_from("<I", data, name_rva)
            name = data[name_rva + 4:name_rva + 4 + nlen].decode("utf-16-le")
            modules.append((base, size, os.path.basename(name)))
    if STREAM_EXCEPTION not in streams:
        sys.exit("dump has no exception stream (guest-fault dump): install cdb for thread stacks")
    _s, rva = streams[STREAM_EXCEPTION]
    tid, _pad = struct.unpack_from("<II", data, rva)
    code, _fl, _rec, addr, nparams = struct.unpack_from("<IIQQI", data, rva + 8)
    params = struct.unpack_from("<15Q", data, rva + 8 + 32)
    ctx_size, ctx_rva = struct.unpack_from("<II", data, rva + 8 + 32 + 120)
    rsp = struct.unpack_from("<Q", data, ctx_rva + 0x98)[0]
    rip = struct.unpack_from("<Q", data, ctx_rva + 0xF8)[0]

    def loc(a):
        for base, size, name in modules:
            if base <= a < base + size:
                return "%s+0x%x" % (name, a - base)
        return None

    print("exception 0x%08x at 0x%x (%s) thread %d" % (code, addr, loc(addr) or "?", tid))
    if nparams >= 2:
        print("  access violation info: kind=%d address=0x%x" % (params[0], params[1]))
    print("rip 0x%x  rsp 0x%x" % (rip, rsp))
    # Stack memory: the thread list entry holds the stack range captured.
    mem = None
    if STREAM_THREADS in streams:
        _s, trva = streams[STREAM_THREADS]
        (n,) = struct.unpack_from("<I", data, trva)
        for i in range(n):
            off = trva + 4 + 48 * i
            (t,) = struct.unpack_from("<I", data, off)
            if t == tid:
                start, size, mrva = struct.unpack_from("<QII", data, off + 24)
                mem = (start, size, mrva)
    if not mem or not (mem[0] <= rsp < mem[0] + mem[1]):
        print("stack memory not in dump")
        return
    start, size, mrva = mem
    print("candidate return addresses (module+offset):")
    shown = 0
    for a in range(rsp - start, size - 7, 8):
        (w,) = struct.unpack_from("<Q", data, mrva + a)
        l = loc(w)
        if l:
            print("  [rsp+0x%x] 0x%x %s" % (a - (rsp - start), w, l))
            shown += 1
            if shown >= 40:
                break
    print("Resolve offsets with the harness PDB: cdb -z <dump> -c '.sympath+ <pdb dir>;.ecxr;kn;q'")


if __name__ == "__main__":
    main()
