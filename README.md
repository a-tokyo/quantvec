<div align="center">

# quantvec

**Data-oblivious, zero-training vector quantization & nearest-neighbor search for TypeScript.**

A clean-room implementation of Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874)
(Zandieh, Daliri, Hadian, Mirrokni — 2025), with the
[RaBitQ](https://arxiv.org/abs/2405.12497) (Gao & Long, SIGMOD 2024) unbiased-estimator correction.
Runs anywhere JavaScript runs — Node, browsers, Bun, Cloudflare Workers, React Native.

<a href="https://npmjs.com/package/quantvec"><img src="https://img.shields.io/npm/v/quantvec.svg"></a> <a href="https://npmjs.com/package/quantvec"></img><img src="https://img.shields.io/npm/dt/quantvec.svg"></img></a> [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE) <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>

**Add vectors, search instantly — no training, no native build, no server.**  
7.9–15.7× smaller than float32. WASM v128 FastScan for up to ~9× faster queries.  
Node · Browser · Bun · Cloudflare Workers · React Native.

</div>

---

## Why quantvec

Most vector quantizers need a **training phase** — k-means codebooks, learned rotations — awkward
when you can't run k-means in-process or ship a trained model. TurboQuant is **data-oblivious**: a
random rotation makes every coordinate follow a known Beta distribution, so the MSE-optimal
scalar codebook is fully determined by `(dim, bits)` with **no data and ~zero indexing time**.

| Feature            | quantvec                                                                                |
| ------------------ | --------------------------------------------------------------------------------------- |
| Training required  | **No** — rotation + codebook fixed by (dim, bits, seed)                                 |
| Compression        | **7.9–15.7×** (true 2/3/4-bit packing)                                                  |
| Query acceleration | **WASM v128 FastScan** — up to ~9× faster than scalar (scales with n), pure-TS fallback |
| Runtimes           | Node · Browser · Bun · Cloudflare Workers · React Native                                |
| Metrics            | `cosine` · `dot` · `euclidean` per query                                                |
| Id types           | `number` · `string` · `bigint`                                                          |
| Dependencies       | **Zero** runtime dependencies                                                           |

> **Scope:** quantvec is a _flat quantized index_ — O(n) scan over compact codes (à la FAISS
> `IndexPQFastScan`) — with an opt-in **IVF coarse quantizer** (`ivf: { nlist }`) that probes
> only the nearest cells for sublinear search on large corpora (**64.8× QPS at equal recall**
> measured at 200k vectors, nprobe=8/1024; the gain grows with n).

A 1M × 1536-d corpus (e.g. OpenAI `text-embedding-ada-002`) is **6.1 GB as float32**. At 4 bits
quantvec packs it into **~780 MB** (7.92×); at 2 bits, **~390 MB** (15.67×) — with **94%+
recall@10** measured on real OpenAI embeddings (see [Benchmarks](#benchmarks)).

---

## Install

```bash
npm install quantvec
# bun add quantvec  /  pnpm add quantvec  /  yarn add quantvec
```

---

## Quick start

### Low-level: `TurboQuantIndex`

```ts
import { TurboQuantIndex } from 'quantvec';

const index = new TurboQuantIndex({ dim: 1536, bits: 4, metric: 'cosine' });

// flat Float32Array (m·dim), or number[][] / Float32Array[]
index.add(vectors);

const { indices, scores } = index.search(query, 10);
// indices: Int32Array (slot positions)  ·  scores: Float32Array (metric values)
```

Enable the **v128 FastScan** kernel for faster queries (4-bit only; approximate ranking +
exact rescore of the candidate pool; speedup grows with n — ~1.8× at 10k, ~9.3× at 100k):

```ts
const index = new TurboQuantIndex({ dim: 1536, bits: 4, fastscan: true });
```

For large corpora, enable the **IVF coarse quantizer** — k-means cells are trained from the
first add (needs ≥ `nlist` vectors; ~32·nlist recommended) and queries probe only the nearest
`nprobe` cells (sublinear scan; **64.8× QPS** at equal recall on 200k vectors, nprobe=8/1024):

```ts
const index = new TurboQuantIndex({ dim: 1536, ivf: { nlist: 1024 } });
index.add(corpus); // first batch trains + freezes the cells
index.search(query, 10, { nprobe: 32 }); // per-query recall/speed knob
```

### Stable ids: `IdMapIndex`

```ts
import { IdMapIndex } from 'quantvec';

const db = new IdMapIndex<string>({ dim: 768, bits: 4 });
db.addWithIds(['doc-1', 'doc-2', 'doc-3'], vectors);

const { ids, scores } = db.search(query, 5); // ids: string[], best-first
db.has('doc-2'); // → true
db.remove('doc-2'); // O(1) swap-remove

// Optional allowlist predicate:
db.search(query, 5, { filter: (id) => id !== 'doc-1' });
```

### High-level: `createCollection` (qdrant-inspired)

The ergonomic layer stores payloads alongside vectors and supports a structured filter DSL:

```ts
import { createCollection } from 'quantvec';

type Doc = { title: string; year: number; published: boolean };

const col = createCollection<Doc>({
  vectors: { size: 1536, distance: 'cosine' },
  quantization: { bits: 4 },
});

// Upsert points with payloads
col.upsert([
  { id: 'a', vector: embedA, payload: { title: 'Alpha', year: 2023, published: true } },
  { id: 'b', vector: embedB, payload: { title: 'Beta', year: 2024, published: false } },
]);

// Search with a filter
const hits = col.search(queryVec, {
  limit: 5,
  filter: {
    must: [
      { key: 'published', match: { value: true } },
      { key: 'year', range: { gte: 2023 } },
    ],
  },
});
// hits: Array<{ id, score, payload }>
```

**Filter DSL** — mirrors qdrant semantics:

| Clause     | Condition types                                                                        |
| ---------- | -------------------------------------------------------------------------------------- |
| `must`     | all must match (AND)                                                                   |
| `should`   | at least one must match (OR), or vacuously true when empty                             |
| `must_not` | none may match (NOT)                                                                   |
| Conditions | `{ key, match: { value } }` · `{ key, range: { gt/gte/lt/lte } }` · `{ hasId: [...] }` |

### Persistence

```ts
// Isomorphic — store as Uint8Array anywhere (IndexedDB, fetch, etc.)
const bytes = index.toBytes();
const restored = TurboQuantIndex.fromBytes(bytes);

// Node helpers (quantvec/node subpath)
import { saveIndex, loadIndex, loadIdMapIndex } from 'quantvec/node';
await saveIndex(index, './index.qv');
const idx = await loadIndex('./index.qv');
```

### Typed errors

Every boundary throws a discriminated, code-tagged error:

```ts
import { IndexError } from 'quantvec';

try {
  index.search(query, 10); // throws if index is empty
} catch (e) {
  if (e instanceof IndexError && e.code === 'EMPTY') {
    /* ... */
  }
}
```

Exported error classes: `IndexError` · `IdMapError` · `DeserializeError` · `EncodeError` · `SearchError` · `FilterError`.

---

## How it works

```mermaid
flowchart LR
  V["input vector v"] --> N["normalize<br/>(store ‖v‖)"]
  N --> R["random rotation Q\n(FWHT for pow-2 dims;\ndense otherwise)"]
  R --> B["coords ≈ Beta((d−1)/2,(d−1)/2)"]
  B --> TQ["TQ+ calibration\n(opt-in; per-coord affine)"]
  TQ --> Q["Lloyd-Max quantize\n2 / 3 / 4 bits"]
  Q --> S["RaBitQ scale\n(unbiased ⟨q,v⟩)"]
  S --> DB[("bit-packed codes\n+ scale + norm")]

  query["query q"] --> RQ["rotate q"]
  RQ --> LUT["nibble LUT\n(dim × levels f32)"]
  LUT --> FS["v128 FastScan\n(u8 LUT, u16 acc)\nor exact WASM kernel"]
  DB --> FS
  FS --> RESCORE["exact rescore\ntop-pool"]
  RESCORE --> TOPK["top-k heap"]
```

1. **Normalize** each vector (store its norm for metric reconstruction).
2. **Rotate** — FWHT for power-of-two dims (O(d·log d), ~25× faster build), dense Householder otherwise. The rotation is data-independent, frozen by `(dim, seed)`.
3. **TQ+ calibration** (opt-in) — per-coordinate affine map from a fit on the first add batch; reduces bias on real embeddings.
4. **Lloyd-Max quantize** — MSE-optimal codebook for the Beta marginal; 2, 3, or 4 bits. No training data needed.
5. **RaBitQ scale** per vector — yields an unbiased inner-product estimate at query time.
6. **Search** — rotates the query once, builds a per-query lookup table, then either:
   - **v128 FastScan** (`fastscan: true`): WASM `swizzle`-based SIMD scan of blocked 16-vector tiles → u16 accumulators → rank candidate pool → exact rescore of the pool. Speedup grows with n (**~1.8× at 10k → ~9.3× at 100k vectors**).
   - **Exact WASM kernel** (default): AssemblyScript f64 accumulation, resident codes in linear memory, bit-identical to the scalar oracle.
   - **Pure-TS scalar** (automatic fallback when WASM is unavailable).

See [`docs/research/`](./docs/research/) for distilled paper notes and architecture details.

---

## Benchmarks

### SIFT-small (real dataset)

10k × 128-d vectors · 100 queries · 100-NN L2 ground truth (`npm run bench:real`).
dim=128 is a power of two → FWHT rotation + WASM kernel active.

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS   | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | ----- | ------------ | ----------- |
| 2    | 0.620    | 0.670     | 0.744      | ~269k          | ~1050 | —            | 12.8×       |
| 3    | 0.720    | 0.801     | 0.863      | ~197k          | ~1084 | —            | 9.1×        |
| 4    | 0.860    | 0.888     | 0.928      | ~177k          | ~1152 | **~2055**    | 7.1×        |

### FastScan speedup

FastScan scales with corpus size. Measured on Apple Silicon:

| corpus    | exact WASM | v128 FastScan | speedup   |
| --------- | ---------- | ------------- | --------- |
| 10k vecs  | 1152 QPS   | 2055 QPS      | **1.8×**  |
| 50k vecs  | ~240 QPS   | ~1350 QPS     | **5.7×**  |
| 100k vecs | ~7 QPS     | ~65 QPS       | **~9.3×** |

The gain grows with `n` because the SIMD scan cost scales linearly while the fixed
rescore-pool overhead stays constant. Enable with `fastscan: true` (4-bit only; pure-TS
fallback when WASM is unavailable).

### Synthetic (dataset-free)

`dim=768, n=5000, cosine` · recall vs exact float32 (`npx tsx benchmarks/flat.ts`):

| bits | recall@10 | fastScan QPS | compression |
| ---- | --------- | ------------ | ----------- |
| 2    | 0.625     | —            | 15.4×       |
| 3    | 0.794     | —            | 10.4×       |
| 4    | 0.887     | **~528**     | 7.8×        |

True bit-packing — on par with native TurboQuant (~15.8× @ 2-bit / ~8.0× @ 4-bit).

### GloVe-200 (real text embeddings)

100k × 200-d word vectors · 1000 queries · brute-force cosine ground truth (`npm run bench:glove`).
dim=200 is not a power of two → exercises the **dense Householder** rotation path.

| bits | recall@1 | recall@10 | recall@100 | QPS | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | --- | ------------ | ----------- |
| 2    | 0.550    | 0.610     | 0.653      | ~69 | —            | 13.8×       |
| 3    | 0.730    | 0.781     | 0.814      | ~72 | —            | 9.6×        |
| 4    | 0.845    | **0.880** | 0.901      | ~71 | **~456**     | 7.4×        |

Real embedding structure consistently lifts recall above the synthetic isotropic floor.
Full results and JSON in [`benchmarks/`](./benchmarks/).

### dbpedia-OpenAI-100k (real text embeddings)

Full 100k × 1536-d OpenAI text-embedding-ada-002 vectors · 973 queries · ann-benchmarks
pre-computed cosine ground truth (`npm run bench:openai`). dim=1536 is a power of two → FWHT
rotation + WASM kernel active.

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | --- | ------------ | ----------- |
| 2    | 0.791    | 0.824     | 0.840      | ~461           | ~7  | —            | 15.67×      |
| 3    | 0.891    | 0.899     | 0.912      | ~462           | ~7  | —            | 10.52×      |
| 4    | 0.953    | **0.944** | 0.952      | ~238           | ~7  | **~65**      | 7.92×       |

High dimensionality and FWHT deliver strong recall against ann-benchmarks ground truth — in line
with the TurboQuant paper's reported numbers on real OpenAI embeddings.
Full results and JSON in [`benchmarks/`](./benchmarks/).

---

## Roadmap

| Status | Item                                                                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| ✅     | Core math: rotation, Beta/Lloyd-Max codebooks, encode pipeline, flat nibble-LUT search                                          |
| ✅     | `TurboQuantIndex`, `IdMapIndex`, versioned serialization, Node fs helpers                                                       |
| ✅     | True 2/3/4-bit **bit-packed serialization** (7.9–15.7× compression)                                                             |
| ✅     | **FWHT rotation** for power-of-two dims (O(d·log d), ~25× faster encode)                                                        |
| ✅     | **TQ+ per-coordinate calibration** (opt-in; data-dependent)                                                                     |
| ✅     | **Exact WASM scoring kernel** (AssemblyScript, bit-identical to scalar, ~1.3× query)                                            |
| ✅     | **v128 FastScan kernel** (blocked-nibble swizzle + exact rescore, **up to ~9.3× query**)                                        |
| ✅     | **Ergonomic `createCollection`** with typed payloads and filter DSL                                                             |
| ✅     | Real-dataset benchmarks: SIFT-small + GloVe-200 + dbpedia-OpenAI-100k (results in `benchmarks/results/`)                        |
| ✅     | **IVF / coarse-quantizer** for large corpora (k-means cells, full remove parity, **64.8× QPS** at equal recall on 200k vectors) |

---

## References

- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** — Zandieh, Daliri,
  Hadian, Mirrokni. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (2025).
- **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest
  Neighbor Search** — Gao & Long. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497), SIGMOD 2024.

---

## License

[Apache-2.0](./LICENSE) © Ahmed Tokyo. See [`NOTICE`](./NOTICE).  
quantvec is an independent clean-room implementation and is not affiliated with or endorsed by Google.
