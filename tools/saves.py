#!/usr/bin/env python3
"""List, export and import the native save containers (XGameSave local store).

On-disk layout (runtime/os/xgamesave_local.cpp):
  <root>/<config-id>/local/c-<container>/b-<blob>       blob bytes
  <root>/<config-id>/local/c-<container>/.display-name  optional UTF-8 text
Names use the store encoding: [a-z0-9_-] stay, 'A'-'Z' -> '^a'-'^z', others -> %XX.
<root> defaults to %LOCALAPPDATA%\\playgta6web\\saves. The zip format is in docs/SAVE_ZIP_FORMAT.md.

  saves.py list   [--root DIR]
  saves.py export OUT.zip [--root DIR] [--config-id ID] [--force]
  saves.py import IN.zip  [--root DIR] [--config-id ID] [--force] [--dry-run]

Only c-*/b-*/.display-name files are ever read or written. Import validates the whole zip first,
refuses to overwrite anything without --force, and swaps each container in atomically.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

FORMAT = "pg6-saves"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
DISPLAY_FILE = ".display-name"
MAX_CONTAINER_NAME = 255
MAX_BLOB_NAME = 128
MAX_BLOB_BYTES = 16 * 1024 * 1024
MAX_DISPLAY = 255
MAX_ENCODED = 240
QUOTA_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_ENTRIES = 20000
_counter = [0]


class SaveError(Exception):
    pass


def _plain(c):
    return (97 <= c <= 122) or (48 <= c <= 57) or c in (95, 45)


def encode(name):
    out = []
    for c in name.encode("utf-8"):
        if _plain(c):
            out.append(chr(c))
        elif 65 <= c <= 90:
            out.append("^" + chr(c + 32))
        else:
            out.append("%%%02X" % c)
    return "".join(out)


def decode(text):
    """Inverse of encode; None when text is not the canonical encoding of valid UTF-8."""
    out = bytearray()
    i = 0
    while i < len(text):
        c = text[i]
        if c == "^" and i + 1 < len(text) and "a" <= text[i + 1] <= "z":
            out.append(ord(text[i + 1]) - 32)
            i += 2
        elif c == "%" and re.fullmatch(r"[0-9A-F]{2}", text[i + 1:i + 3] or ""):
            out.append(int(text[i + 1:i + 3], 16))
            i += 3
        elif ord(c) < 128 and _plain(ord(c)):
            out.append(ord(c))
            i += 1
        else:
            return None
    try:
        name = out.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return name if encode(name) == text else None


def valid_name(name, max_bytes):
    if not name or len(name.encode("utf-8")) > max_bytes:
        return False
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        return False
    return len(encode(name)) <= MAX_ENCODED


def default_root():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
    elif sys.platform == "darwin":
        h = os.environ.get("HOME")
        base = h and os.path.join(h, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME")
        if not base and os.environ.get("HOME"):
            base = os.path.join(os.environ["HOME"], ".local", "share")
    return os.path.join(base, "playgta6web", "saves") if base else "pg6-saves"


def _iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _regular(path):
    return os.path.isfile(path) and not os.path.islink(path)


def scan(root, config_id=None):
    """Returns [{config_id, dir, containers:[...]}], sorted; skips anything that is not a save file."""
    titles = []
    if not os.path.isdir(root):
        return titles
    for cfg_dir in sorted(os.listdir(root)):
        cfg = decode(cfg_dir)
        local = os.path.join(root, cfg_dir, "local")
        if cfg is None or os.path.islink(os.path.join(root, cfg_dir)) or not os.path.isdir(local):
            continue
        if config_id is not None and cfg != config_id:
            continue
        containers = []
        for cdir in sorted(os.listdir(local)):
            cpath = os.path.join(local, cdir)
            if not cdir.startswith("c-") or os.path.islink(cpath) or not os.path.isdir(cpath):
                continue
            cname = decode(cdir[2:])
            if cname is None or not valid_name(cname, MAX_CONTAINER_NAME):
                continue
            blobs, newest = [], 0.0
            for f in sorted(os.listdir(cpath)):
                fp = os.path.join(cpath, f)
                if not f.startswith("b-") or not _regular(fp):
                    continue
                bname = decode(f[2:])
                if bname is None or not valid_name(bname, MAX_BLOB_NAME):
                    continue
                st = os.stat(fp)
                newest = max(newest, st.st_mtime)
                blobs.append({"name": bname, "file": f, "size": st.st_size,
                              "modified": _iso(st.st_mtime), "path": fp})
            display = ""
            dp = os.path.join(cpath, DISPLAY_FILE)
            if _regular(dp) and os.path.getsize(dp) <= MAX_DISPLAY:
                with open(dp, "rb") as fh:
                    display = fh.read().decode("utf-8", "replace")
            containers.append({"name": cname, "dir": cdir, "display_name": display,
                               "modified": _iso(newest) if blobs else None, "blobs": blobs, "path": cpath})
        if containers:
            titles.append({"config_id": cfg, "dir": cfg_dir, "containers": containers})
    return titles


def cmd_list(args):
    titles = scan(args.root)
    if not titles:
        print("no saves under %s" % args.root)
        return 0
    for t in titles:
        print("title %s" % t["config_id"])
        for c in t["containers"]:
            total = sum(b["size"] for b in c["blobs"])
            disp = " (%s)" % c["display_name"] if c["display_name"] else ""
            print("  %s%s: %d blobs, %d bytes, modified %s"
                  % (c["name"], disp, len(c["blobs"]), total, c["modified"] or "-"))
    return 0


def cmd_export(args):
    titles = scan(args.root, args.config_id)
    if not titles:
        raise SaveError("nothing to export under %s" % args.root)
    if os.path.exists(args.out) and not args.force:
        raise SaveError("%s exists (use --force to overwrite)" % args.out)
    man_titles, files = [], []
    for t in titles:
        mc = []
        for c in t["containers"]:
            mb = []
            for b in c["blobs"]:
                with open(b["path"], "rb") as fh:
                    data = fh.read()
                if len(data) > MAX_BLOB_BYTES:
                    raise SaveError("blob %s/%s exceeds %d bytes" % (c["name"], b["name"], MAX_BLOB_BYTES))
                arc = "saves/%s/%s/%s" % (t["dir"], c["dir"], b["file"])
                files.append((arc, data, os.stat(b["path"]).st_mtime))
                mb.append({"name": b["name"], "file": arc, "size": len(data),
                           "sha256": _sha(data), "modified": b["modified"]})
            entry = {"name": c["name"], "modified": c["modified"], "blobs": mb}
            if c["display_name"]:
                data = c["display_name"].encode("utf-8")
                arc = "saves/%s/%s/%s" % (t["dir"], c["dir"], DISPLAY_FILE)
                files.append((arc, data, os.stat(os.path.join(c["path"], DISPLAY_FILE)).st_mtime))
                entry["display_name"] = c["display_name"]
                entry["display_name_file"] = arc
            mc.append(entry)
        man_titles.append({"config_id": t["config_id"], "containers": mc})
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION,
                "created": _iso(datetime.datetime.now().timestamp()),
                "source": "native", "titles": man_titles}
    tmp = args.out + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
        for arc, data, mt in files:
            ts = max(mt, 315532800)
            zi = zipfile.ZipInfo(arc, datetime.datetime.fromtimestamp(ts).timetuple()[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o100644 << 16
            z.writestr(zi, data)
    os.replace(tmp, args.out)
    print("exported %d containers (%d files) to %s"
          % (sum(len(t["containers"]) for t in man_titles), len(files), args.out))
    return 0


def _safe_arc(name):
    if name.startswith("/") or "\\" in name or ".." in name.split("/") or ":" in name or "\x00" in name:
        raise SaveError("unsafe path in zip: %r" % name)


def read_zip(path):
    """Validates a save zip; returns [(config_id, container, display_name, {blob_name: bytes})]."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise SaveError("not a readable zip: %s" % e)
    with z:
        infos = z.infolist()
        if len(infos) > MAX_ENTRIES:
            raise SaveError("too many zip entries")
        by_name = {}
        for zi in infos:
            _safe_arc(zi.filename)
            if zi.filename.endswith("/"):
                continue
            if zi.filename in by_name:
                raise SaveError("duplicate zip entry %s" % zi.filename)
            if (zi.external_attr >> 16) & 0o170000 == 0o120000:
                raise SaveError("symlink entry %s" % zi.filename)
            by_name[zi.filename] = zi
        mi = by_name.get(MANIFEST)
        if mi is None or mi.file_size > MAX_MANIFEST_BYTES:
            raise SaveError("missing or oversized %s" % MANIFEST)
        try:
            man = json.loads(z.read(mi).decode("utf-8"))
        except (ValueError, zipfile.BadZipFile) as e:
            raise SaveError("bad manifest: %s" % e)
        if not isinstance(man, dict) or man.get("format") != FORMAT:
            raise SaveError("not a %s archive" % FORMAT)
        if man.get("format_version") != FORMAT_VERSION:
            raise SaveError("unsupported format_version %r (this tool reads %d)"
                            % (man.get("format_version"), FORMAT_VERSION))
        used, out, seen_cont = set(), [], set()
        for t in man.get("titles") or []:
            cfg = t.get("config_id") if isinstance(t, dict) else None
            if not isinstance(cfg, str) or not cfg or len(encode(cfg)) > MAX_ENCODED:
                raise SaveError("bad title config_id %r" % (cfg,))
            title_bytes = 0
            for c in t.get("containers") or []:
                name = c.get("name") if isinstance(c, dict) else None
                if not isinstance(name, str) or not valid_name(name, MAX_CONTAINER_NAME):
                    raise SaveError("bad container name %r" % (name,))
                if (cfg, name) in seen_cont:
                    raise SaveError("duplicate container %s/%s" % (cfg, name))
                seen_cont.add((cfg, name))
                base = "saves/%s/c-%s/" % (encode(cfg), encode(name))
                blobs = {}
                for b in c.get("blobs") or []:
                    bn = b.get("name") if isinstance(b, dict) else None
                    if not isinstance(bn, str) or not valid_name(bn, MAX_BLOB_NAME) or bn in blobs:
                        raise SaveError("bad or duplicate blob name %r in %s" % (bn, name))
                    arc = base + "b-" + encode(bn)
                    if b.get("file") != arc or arc not in by_name:
                        raise SaveError("blob %s/%s: file entry missing or misnamed" % (name, bn))
                    zi = by_name[arc]
                    if zi.file_size > MAX_BLOB_BYTES or b.get("size") != zi.file_size:
                        raise SaveError("blob %s/%s: size mismatch or too large" % (name, bn))
                    data = z.read(zi)
                    if len(data) != b["size"] or _sha(data) != b.get("sha256"):
                        raise SaveError("blob %s/%s: checksum mismatch" % (name, bn))
                    used.add(arc)
                    blobs[bn] = data
                    title_bytes += len(data)
                disp = c.get("display_name", "")
                if not isinstance(disp, str) or len(disp.encode("utf-8")) > MAX_DISPLAY:
                    raise SaveError("bad display_name in %s" % name)
                if disp:
                    darc = base + DISPLAY_FILE
                    if c.get("display_name_file") != darc or darc not in by_name:
                        raise SaveError("display name file missing for %s" % name)
                    if z.read(by_name[darc]) != disp.encode("utf-8"):
                        raise SaveError("display name mismatch for %s" % name)
                    used.add(darc)
                out.append((cfg, name, disp, blobs))
            if title_bytes > QUOTA_BYTES:
                raise SaveError("title %s exceeds the %d byte quota" % (cfg, QUOTA_BYTES))
        extra = set(by_name) - used - {MANIFEST}
        if extra:
            raise SaveError("unlisted entries in zip: %s" % ", ".join(sorted(extra)[:5]))
        return out


def _write_file(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _container_bytes(cpath):
    total = 0
    if os.path.isdir(cpath):
        for f in os.listdir(cpath):
            fp = os.path.join(cpath, f)
            if f.startswith("b-") and _regular(fp):
                total += os.path.getsize(fp)
    return total


def _title_used(local):
    if not os.path.isdir(local):
        return 0
    return sum(_container_bytes(os.path.join(local, cd)) for cd in os.listdir(local) if cd.startswith("c-"))


def cmd_import(args):
    items = read_zip(args.zip)
    if args.config_id is not None:
        items = [i for i in items if i[0] == args.config_id]
    if not items:
        raise SaveError("no matching containers in %s" % args.zip)
    root = args.root

    def dest_of(cfg, name):
        return os.path.join(root, encode(cfg), "local", "c-" + encode(name))

    conflicts = [(c, n) for c, n, _, _ in items if os.path.exists(dest_of(c, n))]
    if conflicts and not args.force:
        raise SaveError("would overwrite existing containers (use --force): "
                        + ", ".join("%s/%s" % x for x in conflicts))
    for cfg in sorted({i[0] for i in items}):
        local = os.path.join(root, encode(cfg), "local")
        keep = _title_used(local) - sum(_container_bytes(dest_of(c, n)) for c, n, _, _ in items if c == cfg)
        incoming = sum(len(d) for c, _, _, bl in items if c == cfg for d in bl.values())
        if keep + incoming > QUOTA_BYTES:
            raise SaveError("title %s would exceed the %d byte quota" % (cfg, QUOTA_BYTES))
    if args.dry_run:
        for c, n, _, bl in items:
            print("would import %s/%s (%d blobs)%s" % (c, n, len(bl), " [replace]" if (c, n) in conflicts else ""))
        return 0
    for cfg, name, disp, blobs in items:
        local = os.path.join(root, encode(cfg), "local")
        os.makedirs(local, exist_ok=True)
        base = "c-" + encode(name)
        _counter[0] += 1
        tag = "-%d%d" % (os.getpid(), _counter[0])
        fresh = os.path.join(local, ".new-" + base + tag)  # same names the runtime's crash recovery knows
        aside = os.path.join(local, ".old-" + base + tag)
        dest = os.path.join(local, base)
        shutil.rmtree(fresh, ignore_errors=True)
        os.mkdir(fresh)
        try:
            for bn, data in blobs.items():
                _write_file(os.path.join(fresh, "b-" + encode(bn)), data)
            if disp:
                _write_file(os.path.join(fresh, DISPLAY_FILE), disp.encode("utf-8"))
            had = os.path.isdir(dest)
            if had:
                os.rename(dest, aside)
            try:
                os.rename(fresh, dest)
            except OSError:
                if had:
                    os.rename(aside, dest)
                raise
            if had:
                shutil.rmtree(aside, ignore_errors=True)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
        print("imported %s/%s (%d blobs)" % (cfg, name, len(blobs)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="List, export and import native save containers.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "export", "import"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=default_root())
        if name == "export":
            s.add_argument("out")
            s.add_argument("--config-id")
            s.add_argument("--force", action="store_true")
        if name == "import":
            s.add_argument("zip")
            s.add_argument("--config-id")
            s.add_argument("--force", action="store_true")
            s.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        return {"list": cmd_list, "export": cmd_export, "import": cmd_import}[args.cmd](args)
    except SaveError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
