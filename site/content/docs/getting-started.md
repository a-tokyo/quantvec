# Getting Started

quantvec is a **data-oblivious, zero-training** vector quantizer and flat nearest-neighbor index.
There is no fit/train step: the rotation and codebook are fixed by `(dim, bits, seed)`, so you add
vectors and search immediately.

## Install

```bash
npm install quantvec   # or: bun add quantvec / pnpm add quantvec
```

quantvec is ESM/CJS, fully typed, and dependency-free. The main entry is isomorphic (no `node:*`
imports); Node-only filesystem helpers live under the `quantvec/node` subpath.

## Your first index

```ts
import { TurboQuantIndex } from 'quantvec';

// dim must be a positive multiple of 8; bits ∈ {2,3,4} (default 4).
const index = new TurboQuantIndex({ dim: 8, bits: 4, metric: 'cosine' });

index.add([
  [8, 8, 0, 0, 0, 0, 0, 0],
  [0, 0, 8, 8, 0, 0, 0, 0],
  [0, 0, 0, 0, 8, 8, 0, 0],
]);

const { indices, scores } = index.search([8, 8, 0, 0, 0, 0, 0, 0], 2);
console.log(indices[0]); // 0  — the matching vector's slot
```

`add` accepts a flat `Float32Array` of `m·dim` values, or an array of `number[]` / `Float32Array`.
`search` returns `{ indices: Int32Array, scores: Float32Array }`, best-first.

## Searching by your own id

If you hold external ids (document ids, UUIDs, …), use `IdMapIndex` so deletes and results speak in
your ids rather than positional slots:

```ts
import { IdMapIndex } from 'quantvec';

const db = new IdMapIndex<string>({ dim: 8 });
db.addWithIds(['a', 'b', 'c'], vectors);
const { ids, scores } = db.search(query, 3); // ids: string[]
db.remove('b'); // O(1)
```

## Choosing `bits`

| bits | use when |
| ---- | -------- |
| 2    | maximum compression, recall-tolerant workloads |
| 3    | balanced |
| 4    | best recall (default) |

Higher dimensions quantize better — the random rotation makes coordinates near-Gaussian and the
unbiased-estimator variance shrinks with `d`. See [Benchmarks](/docs/benchmarks).

## WASM FastScan (faster queries)

Enable the v128 FastScan kernel for ~2–6× faster queries (4-bit only; approximate SIMD ranking + exact rescore):

```ts
const index = new TurboQuantIndex({ dim: 1536, bits: 4, fastscan: true });
```

FastScan is ignored and falls back to the exact WASM kernel when `bits ≠ 4` or WebAssembly is unavailable.

## Collections (payloads + filters)

`createCollection` is the highest-level API — store typed payloads alongside vectors and filter at search time:

```ts
import { createCollection } from 'quantvec';

const col = createCollection<{ tag: string }>({
  vectors: { size: 768, distance: 'cosine' },
  quantization: { bits: 4 },
});

col.upsert([
  { id: 'doc-1', vector: v1, payload: { tag: 'news' } },
  { id: 'doc-2', vector: v2, payload: { tag: 'blog' } },
]);

const hits = col.search(query, {
  limit: 5,
  filter: { must: [{ key: 'tag', match: { value: 'news' } }] },
});
// hits: { id, score, payload }[]
```

See the [Usage Guide](/docs/guide) for the full filter DSL.

## Next steps

- [Usage Guide](/docs/guide) — metrics, filtering, persistence, error handling.
- [API Reference](/docs/api-reference) — every option and return type.
- [Architecture](/docs/architecture) — how the pipeline works.
