# BCPack texture decompression (DirectStorage)

The title loads many textures through DirectStorage requests whose options carry a **BcpackMode**
(`DSTORAGE_REQUEST_OPTIONS` bits 1-4; the title uses `0x20000e`, i.e. BC7 into physical pages).
BCPack is the console's lossless packing of BC block streams. The decoder lives in
`runtime/os/bcpack.{h,cpp}` and is used by the XCRDAPI queue in `runtime/os/xbox_scarlett_guest.cpp`.

| Mode | Format | Status |
|------|--------|--------|
| 1 | BC1 | decoded |
| 3 | BC3 | decoded |
| 4 | BC4 | decoded |
| 5 | BC5 | decoded |
| 7 | BC7 | decoded |
| 2, 6 | BC2, BC6H | not implemented: E_NOTIMPL |

Decoded blocks go to the request's destination, whether that is a contiguous buffer or the
physical-page array of `DestinationIsPhysicalPages` (bit 21). When zlib (bit 0) is also set, the
source is inflated first into `uncompressed_size` bytes (the intermediate size), then unpacked
into `destination_size` bytes.

## The format tables are user-supplied. We do not ship them.

The decoder needs the format's constant tables: rANS models, lane layouts, index rank tables, BC7
control words and per-format dictionaries. These tables have no public, documented source.
Because of that:

- They are **never committed** to this repository, in any form (binary, generated source,
  test fixtures).
- **No tool in this repository creates them** from game files, SDKs, or anything else.
- You provide them as a file at run time:

  ```
  playgta6_boot ... --bcpack-tables <file>
  # or
  PLAYGTA6_BCPACK_TABLES=<file> playgta6_boot ...
  ```

  If `--bcpack-tables` names a bad file, the harness stops with exit code 64. A bad
  `PLAYGTA6_BCPACK_TABLES` file only prints a warning.

Without tables, BCPack requests keep failing truthfully with `E_NOTIMPL`. The log line says how
to supply the tables. If the tables lack a format, requests for that format fail the same way.
A stream that does not decode fails with `ERROR_INVALID_DATA`.

The boot report counters `dstorage_bcpack_requests` (all BCPack requests) and
`dstorage_bcpack_decoded` (those unpacked) show how many requests the tables cover.

Keep the file outside the repository (for example under `local-data/`). `.gitignore` also
ignores `*.pg6bcpak`.

## Table file format (version 1)

All integers are little-endian. There is no padding.

### Header (16 bytes)

| Offset | Type | Value |
|--------|------|-------|
| 0 | `char[8]` | magic `PG6BCPAK` |
| 8 | `u32` | file version, `1` |
| 12 | `u32` | section count `N` |

### Section directory

The directory follows the header: `N` entries of 16 bytes each.

| Offset | Type | Meaning |
|--------|------|---------|
| 0 | `char[4]` | tag, e.g. `MODL` |
| 4 | `u32` | section version, `1` |
| 8 | `u32` | offset of the payload from the start of the file |
| 12 | `u32` | payload size in bytes |

The loader handles tags like this:
- Unknown tags are ignored, so later versions can add sections.
- A known tag with another version, a duplicate tag, or a payload past the end of the file makes
  the file invalid.
- `MODL` and `LANE` are required.
- A format is decoded only if it has a `LANE` entry and every section it uses:
  - BC1, BC3, BC4 and BC5 need `RANK` and their own `DICn`.
  - BC7 needs `BC7C` and `DIC7`.

### `MODL`: rANS models (2304 bytes)

The section holds 128 models × 9 `u16` bucket frequencies.

Model `m` belongs to group `g = m / 16` and uses buckets `0 .. g+1`. Buckets past those are
ignored.
- **Bucket 0** holds symbol 0. Slot 0 is never coded, so a bucket-0 frequency of `f0` gives
  symbol 0 the slots `1 .. f0-1`.
- **Bucket `b ≥ 1`** holds the `2^(b-1)` symbols from `2^(b-1)` to `2^b - 1`. Each of them has the
  bucket's frequency, and they are laid out one after another.
- The slots of a model must add up to exactly 65536, and every used bucket must be non-zero.

### `LANE`: lane layouts (4 + 20 × count bytes)

The section starts with a `u32 count`, followed by `count` entries:

| Offset | Type | Meaning |
|--------|------|---------|
| 0 | `u8` | BC format (1, 3, 4, 5, 7; other formats are ignored) |
| 1 | `u8` | lane count (1-16) |
| 2 | `u8` | symbols per lane per block (1 or 2) |
| 3 | `u8` | reserved, 0 |
| 4 | `u8[16]` | width of each lane (3-7); unused entries 0 |

- `lanes × symbols per block` must equal the format's record length: BC1 12, BC3 24, BC4 12,
  BC5 24, BC7 32.
- Lane `j` contributes the record symbols `j × per .. j × per + per - 1` of every block.
- A fragment header byte `h` on a lane of width `w` selects model `(w - (h >> 6)) × 16 + (h & 15)`.

### `RANK`: index rank tables (704 bytes)

The tables follow each other in this order:

| Bytes | Table | Entry |
|-------|-------|-------|
| 256 | BC1 | four 2-bit index residuals of a 2×2 quad: bits 0-1 top-left, 2-3 top-right, 4-5 bottom-left, 6-7 bottom-right |
| 64 | BC3 alpha | two 3-bit residuals of neighbouring pixels: bits 0-2 the first, bits 4-6 the second (looked up by symbol & 63) |
| 256 | BC3 colour | as BC1 |
| 64 | BC4 | as BC3 alpha |
| 64 | BC5 | as BC3 alpha |

### `BC7C`: BC7 control words (1024 bytes)

The section holds 512 `u16` words, indexed by `(s31 & 15) << 5 | (s30 & 31)` from the block
record's last two symbols.

Each word packs these fields:
- bits 0-2: BC7 mode;
- bits 3-4: rotation;
- bit 5: index selection;
- bits 6-7: subset count;
- bits 8-13: partition.

A word of 0 marks an unused code. A block that uses it, or one whose subset count does not match
its mode, is corrupt.

### `DIC1`, `DIC3`, `DIC4`, `DIC5`: plain-format dictionaries

Each section holds 255 entries. Each entry gives the endpoints, then the index values:
- Endpoints are in field units. Colour endpoints are R0 G0 B0 R1 G1 B1 at 5/6/5 bits. Alpha (BC4)
  endpoints are bytes whose top bit is flipped relative to the bytes in the BC block.
- Index values are positions along the endpoint line (0 = first endpoint, maximum = second), not
  BC index codes.

| Tag | Entry | Layout |
|-----|-------|--------|
| `DIC1` | 22 bytes | 6 colour endpoints, 16 two-bit positions |
| `DIC3` | 40 bytes | 6 colour endpoints, 2 alpha endpoints; 16 three-bit alpha positions, then 16 two-bit colour positions |
| `DIC4` | 18 bytes | 2 endpoints, 16 three-bit positions |
| `DIC5` | 36 bytes | 4 endpoints (red pair, green pair), 32 three-bit positions (red block, then green block) |

A block `i` that names reference `r > i` uses dictionary entry `r - i - 1`.

### `DIC7`: BC7 dictionary (255 × 116 bytes)

Each entry holds these fields:
- `u16` control word (as in `BC7C`).
- `3 × 8 u16` endpoint rows, one per subset: R0 R1 G0 G1 B0 B1 A0 A1.
  - Each value is left-aligned to 16 bits at its stored precision.
  - The p-bit is kept as red's lowest bit.
  - A rotation swaps alpha with the rotated channel.
- `u8[2]` index bits of the primary and secondary index sets.
- `2 × 16 i16` index values, signed at their index bits.

## Provenance

The decoder logic was written for this repository from a description of the format:
- The stream has 12 or 16 interleaved rANS lanes, 128 static models and a fixed byte schedule.
- Each block is coded against two back references (the last 256 blocks, or a dictionary).
- The decoded symbols are the residuals of endpoints and indices.

We worked out that description by reading VIctory's GPL-3.0 decoder. We copied no code, constants
or tables from it.

The only tables in the source come from the public BC7 specification: mode bit counts and anchor
indices. The anchor indices are as published in Microsoft's MIT-licensed DirectXTex.

Tests (`tests/test_bcpack.cpp`, and the DirectStorage cases in
`tests/test_xbox_scarlett_guest.cpp`) use only synthetic tables and streams from a test-only
encoder (`tests/bcpack_synthetic.h`). The decoder has **not** been checked against real data in
this repository. A user with real tables can check it locally by booting with `--bcpack-tables` and
comparing `dstorage_bcpack_decoded` and the resulting textures.
