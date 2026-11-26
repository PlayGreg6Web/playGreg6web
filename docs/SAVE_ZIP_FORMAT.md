# Save zip format (`pg6-saves` v1)

Portable archive of the user's own save containers. Written and read by `tools/saves.py` (native)
and meant to be the import/export format of the browser build (worker contract: see issue #663).
It contains only save files: no game bytes, no logs, no caches.

## Layout

```
manifest.json
saves/<config-dir>/c-<container>/b-<blob>        blob bytes, verbatim
saves/<config-dir>/c-<container>/.display-name   optional, UTF-8 display name (<= 255 bytes)
```

`<config-dir>`, `<container>` and `<blob>` use the store name encoding of
`runtime/os/xgamesave_local.cpp`: `[a-z0-9_-]` stay, `A-Z` becomes `^` + lower-case, every other
UTF-8 byte becomes `%XX` (upper-case hex). This is exactly the native on-disk layout
(`<root>/<config-dir>/local/c-*/b-*`), so the browser's OPFS/IndexedDB tree maps 1:1 onto the zip.

## manifest.json

```json
{
  "format": "pg6-saves",
  "format_version": 1,
  "created": "2026-01-01T12:00:00Z",
  "source": "native",
  "titles": [
    {
      "config_id": "<title configuration id, decoded>",
      "containers": [
        {
          "name": "<container name>",
          "modified": "<ISO-8601 UTC, newest blob mtime>",
          "display_name": "<optional>",
          "display_name_file": "saves/.../.display-name",
          "blobs": [
            {"name": "<blob name>", "file": "saves/.../b-...", "size": 123,
             "sha256": "<hex>", "modified": "<ISO-8601 UTC>"}
          ]
        }
      ]
    }
  ]
}
```

`source` is `"native"` or `"browser"` (informational). Unknown extra manifest fields are ignored;
a different `format_version` is rejected.

## Import rules (both sides)

- Reject: unsafe paths (`..`, `\`, absolute, `:`), symlinks, duplicate entries, zip entries not
  listed in the manifest, a `file` that is not the canonical path for its names, size or sha256
  mismatch, blob > 16 MiB, container name > 255 bytes, blob name > 128 bytes, display name > 255
  bytes, control characters in names, more than 256 MiB of blobs per title.
- Validate everything before touching storage.
- Never overwrite an existing container unless the user explicitly forces it; on conflict import
  nothing.
- A container is replaced as a unit: build it next to the old one (`.new-c-<name>-N`), swap
  (`.old-c-<name>-N` aside, rename, delete aside). Never merge blobs into an existing container.
- Browser worker: after a successful import, call the same persistence path as
  `playgta6_storage_take_changed` (mark the container directories changed) so OPFS/IndexedDB stay in sync.

## Tool

```
python tools/saves.py list
python tools/saves.py export saves.zip [--config-id ID] [--force]
python tools/saves.py import saves.zip [--config-id ID] [--force] [--dry-run]
```

`--root DIR` overrides the default `%LOCALAPPDATA%\playgta6\saves` (the runtime's default). `list`/`export` fall
back to the legacy `%LOCALAPPDATA%\playgta6web\saves` when the default holds no saves; to move old saves over run
`python tools/saves.py migrate --from <old> --to <new>` (copies and verifies, never deletes the source).
