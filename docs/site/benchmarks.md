# Benchmarks

## Real dataset — SIFT-small

`npm run bench:real` downloads SIFT-small (10 000 × 128-d, 100 queries, 100-NN L2 ground truth, ~5 MB)
and measures recall against the **dataset's own ground truth**. dim=128 is a power of two, so the FWHT
rotation and WASM kernel are active:

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS   | compression |
| ---- | -------- | --------- | ---------- | -------------- | ----- | ----------- |
| 2    | 0.62     | 0.67      | 0.74       | ~255k          | ~990  | 12.8×       |
| 3    | 0.72     | 0.80      | 0.86       | ~191k          | ~1040 | 9.1×        |
| 4    | 0.86     | 0.89      | 0.93       | ~171k          | ~1080 | 7.1×        |

## Synthetic (dataset-free)

A seeded PRNG generates synthetic embeddings, so the numbers are deterministic across machines:

```bash
bun run benchmarks/flat.ts                       # defaults: dim=768, n=5000, queries=500
DIM=1536 N=4000 NQ=300 bun run benchmarks/flat.ts
ANISOTROPY=1.0 bun run benchmarks/flat.ts         # isotropic worst case
```

It prints a table, emits `METRIC key=value` lines, and writes `benchmarks/results/flat-d<DIM>.json`.

## What's measured

- **recall@{1,10,100}** vs an exact float32 cosine brute-force scan (overlap of approx top-k with the
  true top-k, averaged over queries).
- **encode throughput** (vectors/sec) and **query throughput** (QPS, pure-TS scalar kernel).
- **compression** = float32 bytes ÷ serialized `toBytes()` bytes.

## Representative results

`dim=1536, n=4000, anisotropy=0.3, cosine` (seed 42):

| bits | recall@1 | recall@10 | recall@100 | compression (serialized) |
| ---- | -------- | --------- | ---------- | ------------------------ |
| 2    | 0.55     | 0.64      | 0.73       | 15.7×                    |
| 3    | 0.75     | 0.80      | 0.85       | 10.5×                    |
| 4    | 0.86     | 0.89      | 0.92       | 7.9×                     |

Recall rises with dimension and with `bits`; serialized compression scales inversely with `bits`.

## Honest accounting

- **Recall regime.** Synthetic isotropic Gaussian data is a *worst case* — neighbors are near-tied, so
  a data-oblivious quantizer can't perfectly order them. Real embeddings (and lower `ANISOTROPY`) do
  better, consistent with the TurboQuant paper's **>90% recall@10 at 2–4 bits** on real
  DBpedia/OpenAI/GloVe sets.
- **Serialized compression** (`toBytes`) is true bit-packed 2/3/4-bit — 7.9–15.7×, on par with native
  TurboQuant implementations (~15.8× @ 2-bit, ~8.0× @ 4-bit). In-memory the index still holds one byte
  per code; in-memory packing lands with the WASM-SIMD scan.
- **TQ+ per-coordinate calibration** is opt-in (`calibrate: true`) and **off** in these runs. Measured
  on this synthetic data it is neutral-to-slightly-negative — the random rotation already equalizes
  coordinate variances — so it is not on by default; its recall lift shows on real embeddings (a
  **real-dataset suite** is a [roadmap](/docs/roadmap) item).
- **FWHT rotation** is used automatically for power-of-two dims (these runs use 768/1536, which are not
  powers of two, so they use the dense rotation); at e.g. dim 1024 it encodes ~25× faster, recall-neutral.

Numbers were produced with the pure-TypeScript scalar kernel; the WASM-SIMD kernel (Wave 6) targets the
same recall at higher QPS.
