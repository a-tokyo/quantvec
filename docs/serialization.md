# Serialization Format

`toBytes()` / `fromBytes()` use a single, versioned, little-endian binary format (`QVEC`). It is the
only format quantvec ships (no legacy readers); pre-1.0 changes bump the version and rewrite. An index
is fully reconstructable from `(dim, bits, seed)` — which regenerate the rotation and codebook — so
only the compact per-vector data (and, for the id-keyed index, the ids) is stored.

## Layout

All multi-byte fields are little-endian. The header is 24 bytes:

| Offset | Size | Field   | Notes                                    |
| ------ | ---- | ------- | ---------------------------------------- |
| 0      | 4    | magic   | `"QVEC"` (`0x51 0x56 0x45 0x43`)         |
| 4      | 1    | version | `2`                                      |
| 5      | 1    | kind    | `0` = positional, `1` = id-keyed         |
| 6      | 1    | metric  | `0` = dot, `1` = cosine, `2` = euclidean |
| 7      | 1    | bits    | `2`, `3`, or `4`                         |
| 8      | 4    | dim     | u32, positive multiple of 8              |
| 12     | 4    | n       | u32, live vector count                   |
| 16     | 8    | seed    | f64, RNG seed of the rotation            |

Body, immediately after the header:

```
codes        : ⌈n·dim·bits/8⌉ bytes   (tightly bit-packed, LSB-first; dim is a multiple of 8
                                        so this is exact — no padding waste)
scales       : n · f32                (per-vector RaBitQ scale)
norms        : n · f32                (per-vector ‖v‖)
calibration  : flag u8 ∈ {0,1}        (1 → shift dim·f32 + scale dim·f32)
ivf          : flag u8 ∈ {0,1}        (1 → nlist u32 + nprobe u32
                                            + centroids nlist·dim·f32 + listForSlot n·u32;
                                        posting lists are rebuilt from listForSlot on load)
ids          : n × tagged id          (id-keyed only)
```

Codes are stored at true 2/3/4 bits per coordinate, so the serialized index is 7.9–15.7× smaller than
float32 (on par with native TurboQuant implementations).

Each id is `tag (u8)` then payload:

| tag | type   | payload                                            |
| --- | ------ | -------------------------------------------------- |
| 0   | number | f64                                                |
| 1   | string | u32 length + UTF-8 bytes                           |
| 2   | bigint | u32 length + UTF-8 of the canonical decimal string |

## Untrusted-input hardening

`fromBytes` treats the buffer as **untrusted**. Before any bulk read or allocation it validates:

- the magic and version;
- `kind`, `metric`, and `bits` are known values;
- `dim` is a positive multiple of 8 and `seed` is finite;
- the declared body size fits within the buffer (so a crafted huge `n` can't trigger an out-of-bounds
  read or an out-of-memory allocation — it's rejected first);
- the calibration section: finite shift/scale with no zero scale (a divisor at query time);
- the ivf section: `nlist ∈ [2, 2^22]`, `nprobe ∈ [1, nlist]`, section bounds checked before any
  allocation, finite centroids, and every `listForSlot` entry `< nlist`;
- every id: bounds-checked length, **valid UTF-8** (fatal decode), **canonical** bigint decimal, and
  **no duplicate ids** (a collision would silently break the id↔slot bijection).

Loading the wrong kind (e.g. positional bytes into `IdMapIndex.fromBytes`) throws `WRONG_KIND`; any
structural problem throws a `DeserializeError` with a specific `.code`
(`BAD_MAGIC`, `BAD_VERSION`, `BAD_KIND`, `BAD_METRIC`, `BAD_BITS`, `BAD_DIM`, `BAD_SEED`, `BAD_LENGTH`,
`BAD_ID`, `BAD_CALIBRATION`, `BAD_IVF`, `TOO_SHORT`).

## Compatibility notes

- Version 2 (current) added the ivf presence byte and section. Per the pre-1.0 policy there are no
  legacy readers: v1 buffers are rejected with `BAD_VERSION` — re-serialize with the current library.
- The id type is **not** stored; pass it to `IdMapIndex.fromBytes<Id>` and ensure it matches.
- The on-disk `codes` section is bit-packed; the index still holds one byte per code in memory.
  In-memory packing (and a SIMD scan over packed codes) is on the [roadmap](roadmap.md).
