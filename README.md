<div align="center">

# quantvec

**Data-oblivious, zero-training vector quantization & nearest-neighbor search for TypeScript.**

A clean-room implementation of Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874),
with the [RaBitQ](https://arxiv.org/abs/2405.12497) unbiased-estimator correction.
Runs anywhere JavaScript runs — Node, browsers, Bun, Cloudflare Workers, React Native.

[![npm](https://img.shields.io/npm/v/quantvec.svg)](https://www.npmjs.com/package/quantvec)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

</div>

> [!WARNING]
> **Early development.** The public API below is the target design and is being built wave by
> wave — see [`docs/worklog/PROGRESS.md`](./docs/worklog/PROGRESS.md). Not yet published to npm.

## Why quantvec

Most vector quantizers need a **training phase** (k-means codebooks, learned rotations) — awkward
when you can't ship a trained model or run k-means in-process. TurboQuant is **data-oblivious**: a
random rotation makes every coordinate follow a known Beta distribution, so an optimal per-coordinate
scalar quantizer works with **no training and ~zero indexing time**. That makes quantvec a natural fit
for **edge, serverless, and browser** vector search.

- **Zero training, instant ingest** — add vectors, search immediately. No fit step.
- **2–4 bit compression** — e.g. ~16× smaller than float32 at 4-bit / d=1536.
- **Isomorphic** — one package for Node, browsers, Bun, Workers, RN. Standard ESM/CJS + types.
- **WASM-SIMD core** with a pure-TypeScript fallback, so it's fast where SIMD exists and correct everywhere.
- **Filtered search** — restrict results by an allowlist/mask or a qdrant-style filter DSL.

> **Scope:** quantvec is a _flat quantized index_ — search is a fast linear scan over tiny 2–4-bit
> codes (à la FAISS `IndexPQFastScan`), not an HNSW graph. Excellent recall and great throughput up to
> ~1–10M vectors; a coarse-quantizer/IVF layer for larger corpora is on the roadmap.

## Install

```bash
npm install quantvec   # or: bun add quantvec / pnpm add quantvec
```

## Quick start (target API)

```ts
import { TurboQuantIndex } from 'quantvec';

const index = new TurboQuantIndex({ dim: 1536, bits: 4 });
index.add(vectors); // Float32Array | number[][], no training
const { indices, scores } = index.search(query, 10);
```

Ergonomic, qdrant-style layer with payloads and filtering:

```ts
import { createCollection } from 'quantvec';

const c = createCollection({
  vectors: { size: 1536, distance: 'cosine' },
  quantization: { bits: 4 },
});
c.upsert([{ id: 1, vector, payload: { tag: 'docs' } }]);
const hits = c.search(query, {
  limit: 10,
  filter: { must: [{ key: 'tag', match: { value: 'docs' } }] },
});
```

## How it works

1. **Normalize** each vector (store its norm).
2. **Random rotation** (data-independent) → each coordinate ~ Beta((d−1)/2, (d−1)/2).
3. **Lloyd-Max scalar quantization** — the MSE-optimal codebook for that known distribution.
4. **Bit-pack** to 2/3/4 bits per coordinate.
5. **RaBitQ length-renormalization scale** per vector → an unbiased inner-product estimate at query time.

See [`docs/research/`](./docs/research/) for the distilled paper notes and design.

## Status & roadmap

Tracked in [`docs/worklog/PROGRESS.md`](./docs/worklog/PROGRESS.md). Built with a doer / verifier /
devil's-advocate subagent workflow and benchmarked the way the paper is (recall@k, QPS, compression).

## License

[Apache-2.0](./LICENSE) © Ahmed Tokyo. See [`NOTICE`](./NOTICE). quantvec is an independent clean-room
implementation and is not affiliated with or endorsed by Google.
