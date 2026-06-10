# Usage Guide

An end-to-end tour of the two index classes, metrics, filtering, persistence, and error handling.

## Three layers

- **`TurboQuantIndex`** — positional. A vector's identity is its insertion slot. `swapRemove(i)`
  deletes in O(1) by moving the last row into the gap (so slot numbers can change). Use it when you
  track positions yourself or never delete.
- **`IdMapIndex<Id>`** — stable ids. Wraps a `TurboQuantIndex` and keeps an id↔slot map, so you add,
  search, and remove by _your_ id. `Id` is `number` (default), `string`, or `bigint`.
- **`Collection<P>`** — the ergonomic, qdrant-style layer: payloads + a structured filter DSL on top
  of `IdMapIndex` (see [Collections](#collections-payloads--filters)).

All three layers accept the same scaling knobs: `calibrate` (TQ+), `fastscan` (4-bit SIMD), and
`ivf` (coarse-quantized sublinear search — see [IVF](#ivf-coarse-quantizer)).

## Adding vectors

```ts
// Flat row-major buffer (fastest for bulk ingest):
const flat = new Float32Array(m * dim);
index.add(flat);

// Or arrays of vectors:
index.add([vecA, vecB]); // Float32Array[] or number[][]
index.addOne(vecA); // single vector

// Id-keyed (one id per vector, same order):
db.addWithIds([101, 102], [vecA, vecB]);
```

`IdMapIndex.addWithIds` validates all ids and vector shapes **up front**, so a duplicate id or a
wrong-length vector aborts before anything is added. (A non-finite/zero vector is an encoder error and
can surface mid-batch; see [Errors](#errors).)

## Metrics

Set a default at construction and override per query. Norms are stored, so all three work without
keeping the original vectors:

```ts
const index = new TurboQuantIndex({ dim, metric: 'cosine' });
index.search(query, 10); // cosine (default)
index.search(query, 10, { metric: 'dot' }); // inner product
index.search(query, 10, { metric: 'euclidean' }); // squared L2 (scores are dist²)
```

## Filtering

`TurboQuantIndex.search` takes a positional `mask` (a `Uint8Array | boolean[]` of length `size`;
slot `j` is scanned only if `mask[j]` is truthy):

```ts
const mask = new Uint8Array(index.size).fill(1);
mask[3] = 0; // exclude slot 3
index.search(query, 10, { mask });
```

`IdMapIndex.search` takes a `filter` predicate over your ids (it builds the slot mask for you):

```ts
db.search(query, 10, { filter: (id) => id !== 'archived' });
```

## Persistence

Every index serializes to a single versioned `Uint8Array` and back. The format is runtime-agnostic
(store it anywhere) and the load path validates untrusted input field by field.

```ts
const bytes = index.toBytes();
const restored = TurboQuantIndex.fromBytes(bytes);

const idBytes = db.toBytes();
const restoredDb = IdMapIndex.fromBytes<string>(idBytes); // assert your id type
```

> The id type is **not** stored in the bytes — pass it to `fromBytes<Id>` if you used a non-default
> type, and it must match what you serialized. Loading the wrong kind (positional bytes into
> `IdMapIndex.fromBytes`, or vice-versa) throws `WRONG_KIND`.

In Node, the `quantvec/node` subpath wraps this with the filesystem:

```ts
import { saveIndex, loadIndex, loadIdMapIndex, readIndexBytes } from 'quantvec/node';

await saveIndex(index, './index.qv');
const idx = await loadIndex('./index.qv'); // TurboQuantIndex
const db = await loadIdMapIndex('./db.qv'); // IdMapIndex
```

In the browser / Workers, persist `toBytes()` to IndexedDB, Cache API, KV, or `fetch` it back.

## Errors

All boundaries throw discriminated, code-tagged errors (never a raw `TypeError`). Switch on `.code`:

```ts
import { IndexError, IdMapError, DeserializeError } from 'quantvec';

try {
  db.remove('missing');
} catch (e) {
  if (e instanceof IdMapError && e.code === 'UNKNOWN_ID') {
    /* ... */
  }
}
```

| Error              | Sample codes                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `IndexError`       | `INVALID_DIM`, `INVALID_BITS`, `INVALID_SEED`, `INVALID_VECTOR`, `INVALID_LENGTH`, `INVALID_INDEX`, `INVALID_NLIST`, `INVALID_NPROBE`, `EMPTY`, `WRONG_KIND` |
| `IdMapError`       | `DUPLICATE_ID`, `UNKNOWN_ID`, `COUNT_MISMATCH`, `INVALID_ID_TYPE`, `INVALID_VECTOR`, `EMPTY`, `WRONG_KIND`                                                   |
| `DeserializeError` | `BAD_MAGIC`, `BAD_VERSION`, `BAD_KIND`, `BAD_DIM`, `BAD_SEED`, `BAD_LENGTH`, `BAD_ID`, `BAD_IVF`, `TOO_SHORT`                                                |
| `EncodeError`      | `ZERO_VECTOR`, `INVALID_LENGTH`, `DEGENERATE`                                                                                                                |
| `SearchError`      | `INVALID_K`, `ZERO_QUERY`, `INVALID_MASK`, `INVALID_SLOT`                                                                                                    |

## Calibration (TQ+)

quantvec can fit an optional **per-coordinate calibration** (the TurboQuant+ refinement) from the first
add of at least 1000 vectors, freeze it for the index's lifetime, and serialize it. It is **opt-in**
(`{ calibrate: true }`); `index.calibrated` reports whether it's active.

```ts
const idx = new TurboQuantIndex({ dim: 768, calibrate: true });
idx.add(firstBatch); // ≥ 1000 vectors → calibration is fit and frozen
idx.calibrated; // → true
```

Calibration remaps each rotated coordinate onto the canonical marginal. It is **data-dependent**: it
can lift recall on real embeddings (the paper's regime) but is neutral-to-slightly-negative on
well-conditioned data where the random rotation already yields near-canonical coordinates — so it is
off by default. Validate a recall gain on your own data before enabling it. It costs only two
`dim`-length vectors in the serialized index.

Because the calibration is frozen from the first batch, a later vector that lies far outside that
distribution (e.g. anti-correlated with a tight calibration cluster) may not be encodable faithfully;
`add` rejects it with `EncodeError` code `DEGENERATE`. If your data drifts that far, rebuild the index
without `calibrate`.

## IVF (coarse quantizer)

For large corpora the O(n) flat scan becomes the bottleneck. Enabling `ivf` partitions the corpus
into `nlist` k-means cells; each query ranks the cell centroids and scans only the `nprobe` nearest
cells:

```ts
const idx = new TurboQuantIndex({ dim: 768, ivf: { nlist: 256 } });
idx.add(corpus); // first add of ≥ nlist vectors trains + freezes the cells
idx.ivfActive; // → true
idx.search(q, 10); // probes ⌈nlist/8⌉ cells by default
idx.search(q, 10, { nprobe: 64 }); // recall/speed knob, per query
```

- **Training**: the cells are fit (seeded k-means++, spherical for cosine/dot, L2 for euclidean)
  from the **first** non-empty add and frozen for the index's lifetime — exactly the calibration
  contract. The hard minimum is `nlist` vectors (≥ ~32·nlist recommended); a smaller first batch
  freezes the index flat forever. Choose `nlist ≈ √n` as a starting point.
- **Exactness**: the probed-cell scan uses the same exact scalar kernel as the flat path, so
  `nprobe = nlist` reproduces the flat scan bit-for-bit; smaller `nprobe` trades recall for speed
  (measured: ~11× QPS at the flat scan's recall with `nprobe = nlist/16` on clustered data).
- **Mutations**: `add`/`addOne` assign new vectors to their nearest cell; `swapRemove`/`remove`
  keep the posting lists in lockstep (full parity with the flat index); `clear()` keeps the trained
  cells. Serialization round-trips the whole structure (format v2).
- **Trade-offs**: while IVF is active the whole-database WASM/FastScan kernels are bypassed (a
  cell-resident kernel is a future wave), and `calibrate`'s `DEGENERATE` caveat applies to cell
  quality too: heavy data drift after training degrades the partition — rebuild to retrain.

The same knobs flow through the other layers: `new IdMapIndex({ dim, ivf: { nlist } })` and
`createCollection({ ..., ivf: { nlist } })`, with `nprobe` accepted by their search options.

## Collections (payloads + filters)

`createCollection` is the highest-level API — store points with typed payloads and query with a
qdrant-style filter:

```ts
import { createCollection } from 'quantvec';

const c = createCollection<{ tag: string; year: number }>({
  vectors: { size: 768, distance: 'cosine' },
  quantization: { bits: 4 },
});

c.upsert([
  { id: 'a', vector: vecA, payload: { tag: 'docs', year: 2024 } },
  { id: 'b', vector: vecB, payload: { tag: 'blog', year: 2022 } },
]);

const hits = c.search(query, {
  limit: 10,
  filter: {
    must: [{ key: 'tag', match: { value: 'docs' } }],
    should: [{ key: 'year', range: { gte: 2023 } }],
    must_not: [{ hasId: ['a'] }],
  },
});
// hits: { id, score, payload }[] — best-first, only points passing the filter
```

Filter DSL: `must` (AND) / `should` (OR, ≥1) / `must_not` (NONE) of leaf conditions —
`{ key, match: { value } }`, `{ key, range: { gt?, gte?, lt?, lte? } }`, `{ hasId: [...] }` — or a
nested filter. `upsert` is insert-or-replace by id; `delete(id | ids)` removes; `get(id)` returns the
payload; pass `withPayload: false` to omit payloads from hits.

## Lifecycle helpers

```ts
index.size; // live vector count
index.clear(); // drop all vectors (keeps capacity)
db.ids(); // snapshot of all ids in slot order
db.has(id); // membership test
```
