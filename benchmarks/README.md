# quantvec benchmarks

Two harnesses: a **synthetic** one (`flat.ts`, deterministic, no download) and a
**real-dataset** one (`real.ts`, SIFT-small with the dataset's own ground truth).

## Real dataset (SIFT-small)

```bash
npm run bench:real   # downloads SIFT-small (~5 MB) then runs; or:
node benchmarks/download-siftsmall.mjs && npx tsx benchmarks/real.ts
```

SIFT-small: 10 000 base × 128-d, 100 queries, 100-NN L2 ground truth (recall is vs that
ground truth, not a self-computed one). dim=128 is a power of two → the FWHT rotation +
WASM kernel are active.

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS   | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | ----- | ------------ | ----------- |
| 2    | 0.620    | 0.670     | 0.744      | ~269k          | ~1050 | —            | 12.8×       |
| 3    | 0.720    | 0.801     | 0.863      | ~197k          | ~1084 | —            | 9.1×        |
| 4    | 0.860    | 0.888     | 0.928      | ~177k          | ~1152 | **~2055**    | 7.1×        |

Encode throughput is fast because dim=128 is a power of two → FWHT rotation (O(d·log d)).
Compression is slightly lower than the high-dim case since the per-vector norm+scale
overhead is relatively larger at d=128.

Full results: [`results/sift-small.json`](./results/sift-small.json).

## Synthetic (flat)

```bash
npx tsx benchmarks/flat.ts                       # defaults: dim=768, n=5000, queries=500
DIM=1536 N=4000 NQ=300 npx tsx benchmarks/flat.ts
ANISOTROPY=1.0 npx tsx benchmarks/flat.ts        # isotropic worst case
```

Env knobs: `DIM`, `N`, `NQ`, `ANISOTROPY` (1.0 = isotropic Gaussian; lower = more
anisotropic, closer to real embedding spectra). The script prints a table, emits
`METRIC key=value` lines for the autoresearch loop, and writes
`benchmarks/results/flat-d<DIM>.json`.

Latest run (`dim=768, n=5000, queries=500, anisotropy=0.3`, cosine):

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS  | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | ---- | ------------ | ----------- |
| 2    | 0.476    | 0.625     | 0.717      | ~1913          | ~259 | —            | 15.4×       |
| 3    | 0.700    | 0.794     | 0.847      | ~1868          | ~263 | —            | 10.4×       |
| 4    | 0.850    | 0.887     | 0.923      | ~1831          | ~263 | **~528**     | 7.8×        |

Full results: [`results/flat-d768.json`](./results/flat-d768.json).

### FastScan

FastScan (`fastscan: true`) is a v128 SIMD scan path, **4-bit only**, shipped in the current
release. It stores codes in a blocked 16-vector layout, performs a WASM `swizzle`-based table
lookup per coordinate, accumulates per-vector u16 sums, ranks a candidate pool, then rescores
the pool exactly. The result is higher throughput at equivalent recall.

```ts
const index = new TurboQuantIndex({ dim: 1536, bits: 4, fastscan: true });
```

The speedup scales with `n` — on 50k × 128-d vectors the gain is **~5.7×**; on SIFT-small
(10k vectors) it is **~1.8×** because the rescore pass is relatively larger. FastScan is
ignored (falls back to the exact scan) when `bits ≠ 4` or WebAssembly is unavailable.

## What is measured

- **recall@{1,10,100}** — fraction of the true top-k returned, averaged over queries.
  Ground truth is the dataset's own (SIFT-small) or exact float32 cosine brute-force (flat).
- **encode throughput** (vectors/sec, single-threaded).
- **QPS** — exact WASM kernel path, single-threaded.
- **fastScan QPS** (4-bit only) — separate measurement pass with `fastscan: true`; `—` for 2/3-bit rows.
- **compression** — float32 bytes ÷ serialized `toBytes()` bytes (true bit-packing: 7.8–15.4×).

## Methodology notes

- **Recall regime.** Isotropic Gaussian data (`ANISOTROPY=1`) is the worst case — no
  neighborhood structure, so the k-th and (k+1)-th true neighbors are near-tied and the
  quantizer cannot perfectly separate them. Real embeddings (and `ANISOTROPY<1`) have a
  power-law spectrum that lifts recall above the isotropic floor, consistent with the
  TurboQuant paper's ">90% recall@10 at 2–4 bits" on DBpedia/OpenAI/GloVe.
- **Serialized compression** (`toBytes`) is true bit-packed 2/3/4-bit — on par with the
  native reference implementation.
- **TQ+ per-coordinate calibration** is available as an opt-in (`calibrate: true`); it
  improves recall on real embeddings with anisotropic structure.
