# quantvec benchmarks

Two harnesses: a **synthetic** one (`flat.ts`, deterministic, no download) and a
**real-dataset** one (`real.ts`, SIFT-small with the dataset's own ground truth).

## Real dataset (SIFT-small)

```bash
npm run bench:real   # downloads SIFT-small (~5 MB) then runs; or:
node benchmarks/download-siftsmall.mjs && bun run benchmarks/real.ts
```

SIFT-small: 10 000 base × 128-d, 100 queries, 100-NN L2 ground truth (recall is vs that
ground truth, not a self-computed one). dim=128 is a power of two → the FWHT rotation +
WASM kernel are active. Representative (`results/sift-small.json`):

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS   | compression |
| ---- | -------- | --------- | ---------- | -------------- | ----- | ----------- |
| 2    | 0.62     | 0.67      | 0.74       | ~255k          | ~990  | 12.8×       |
| 3    | 0.72     | 0.80      | 0.86       | ~191k          | ~1040 | 9.1×        |
| 4    | 0.86     | 0.89      | 0.93       | ~171k          | ~1080 | 7.1×        |

(Encode is fast because FWHT is O(d·log d); compression is a touch lower than the
high-dim case since the per-vector norm+scale overhead is relatively larger at d=128.)

## Synthetic (flat) — Run

```bash
bun run benchmarks/flat.ts                      # defaults: dim=768, n=5000, queries=500
DIM=1536 N=4000 NQ=300 bun run benchmarks/flat.ts
ANISOTROPY=1.0 bun run benchmarks/flat.ts        # isotropic worst case
```

Env knobs: `DIM`, `N`, `NQ`, `ANISOTROPY` (1.0 = isotropic Gaussian; lower = more
anisotropic, closer to real embedding spectra). The script prints a table, emits
`METRIC key=value` lines for the autoresearch loop, and writes
`benchmarks/results/flat-d<DIM>.json`.

## What is measured

- **recall@{1,10,100}** vs an exact float32 cosine brute-force scan (the ground truth).
  recall@k = |approx top-k ∩ exact top-k| / k, averaged over queries.
- **encode throughput** (vectors/sec) and **query throughput** (QPS, pure-TS scalar
  kernel, single-threaded).
- **compression** = float32 bytes ÷ serialized `toBytes()` bytes.

## Methodology notes (honest accounting)

- **Recall regime.** Isotropic Gaussian data is the _worst case_ — it has no
  neighborhood structure, so the 10th and 11th true neighbors are near-tied and a
  data-oblivious quantizer cannot perfectly separate them. Real embeddings (and the
  `ANISOTROPY<1` setting) have a power-law spectrum and do better. These numbers are a
  conservative lower bound, consistent with the TurboQuant paper's ">90% recall@10 at
  2–4 bits" on real DBpedia/OpenAI/GloVe data.
- **Serialized compression** (`toBytes`) is true bit-packed 2/3/4-bit — 7.9–15.7×, on par
  with native TurboQuant implementations. In-memory the index still holds one byte per
  code (a simple, fast scalar kernel); in-memory packing lands with the WASM-SIMD scan.
- **TQ+ per-coordinate calibration** (a further recall lift) is roadmap.

## Latest results (seed 42, cosine)

See `results/*.json`. Representative `dim=1536, n=4000, anisotropy=0.3`:

| bits | recall@1 | recall@10 | recall@100 | compression (serialized) |
| ---- | -------- | --------- | ---------- | ------------------------ |
| 2    | 0.55     | 0.64      | 0.73       | 15.7×                    |
| 3    | 0.75     | 0.80      | 0.85       | 10.5×                    |
| 4    | 0.86     | 0.89      | 0.92       | 7.9×                     |
