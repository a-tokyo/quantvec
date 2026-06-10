# Benchmarks

Four harnesses in `benchmarks/`: synthetic (no download), SIFT-small (~5 MB), GloVe-200 (~918 MB),
and dbpedia-OpenAI-100k (~1.18 GB).

## SIFT-small (real dataset)

`npm run bench:real` — 10k × 128-d vectors, 100 queries, 100-NN L2 ground truth from the dataset.
dim=128 is a power of two → FWHT rotation + WASM kernel active:

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS   | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | ----- | ------------ | ----------- |
| 2    | 0.620    | 0.670     | 0.744      | ~269k          | ~1050 | —            | 12.8×       |
| 3    | 0.720    | 0.801     | 0.863      | ~197k          | ~1084 | —            | 9.1×        |
| 4    | 0.860    | 0.888     | 0.928      | ~177k          | ~1152 | **~2055**    | 7.1×        |

## GloVe-200 (real text embeddings)

`npm run bench:glove` — 100k of 1.18M × 200-d GloVe word vectors, 1000 queries, brute-force cosine
ground truth within the sub-sample. dim=200 is **not** a power of two → dense Householder rotation:

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | --- | ------------ | ----------- |
| 2    | 0.550    | 0.610     | 0.653      | ~27k           | ~69 | —            | 13.8×       |
| 3    | 0.730    | 0.781     | 0.814      | ~20k           | ~72 | —            | 9.6×        |
| 4    | 0.845    | **0.880** | 0.901      | ~19k           | ~71 | **~456**     | 7.4×        |

Encode throughput is lower than SIFT-small because dim=200 uses the O(d²) dense rotation; SIFT-small
uses the O(d·log d) FWHT.

## dbpedia-OpenAI-100k (real text embeddings, 1536-d)

`npm run bench:openai` — full 100k × 1536-d OpenAI text-embedding-ada-002 vectors, 973 queries,
ann-benchmarks pre-computed cosine ground truth (full corpus). dim=1536 is a power of two → FWHT
rotation + WASM kernel active:

| bits | recall@1 | recall@10 | recall@100 | encode (vec/s) | QPS | fastScan QPS | compression |
| ---- | -------- | --------- | ---------- | -------------- | --- | ------------ | ----------- |
| 2    | 0.791    | 0.824     | 0.840      | ~461           | ~7  | —            | 15.67×      |
| 3    | 0.891    | 0.899     | 0.912      | ~462           | ~7  | —            | 10.52×      |
| 4    | 0.953    | 0.944     | 0.952      | ~238           | ~7  | **~65**      | 7.92×       |

High dimensionality and the power-of-two FWHT path deliver strong recall. FastScan speedup is ~9×
at 100k vectors (vs ~1.4× at 5k — the gain grows with n, consistent with the O(n) scan cost).

## FastScan speedup

FastScan (`fastscan: true`, 4-bit only) speedup scales with corpus size:

| corpus    | exact WASM | v128 FastScan | speedup   |
| --------- | ---------- | ------------- | --------- |
| 10k vecs  | ~1152 QPS  | ~2055 QPS     | **1.8×**  |
| 50k vecs  | ~240 QPS   | ~1350 QPS     | **5.7×**  |
| 100k vecs | ~7 QPS     | ~65 QPS       | **~9.3×** |

The SIMD scan cost is O(n) while the rescore-pool overhead is constant, so the gain grows with n.

## Synthetic (dataset-free)

`npx tsx benchmarks/flat.ts` — seeded PRNG, deterministic, no download.
`dim=768, n=5000, queries=500, anisotropy=0.3, cosine`:

| bits | recall@10 | fastScan QPS | compression |
| ---- | --------- | ------------ | ----------- |
| 2    | 0.625     | —            | 15.4×       |
| 3    | 0.794     | —            | 10.4×       |
| 4    | 0.887     | **~528**     | 7.8×        |

## What's measured

- **recall@{1,10,100}** — fraction of the true top-k returned, averaged over queries.
- **encode throughput** (vectors/sec, single-threaded).
- **QPS** — exact WASM kernel path, single-threaded.
- **fastScan QPS** (4-bit only) — separate measurement with `fastscan: true`; `—` for 2/3-bit rows.
- **compression** — float32 bytes ÷ serialized `toBytes()` bytes (true bit-packing).

## Honest accounting

- **Recall regime.** Synthetic isotropic Gaussian data (`ANISOTROPY=1`) is a worst case — neighbors
  are near-tied and a data-oblivious quantizer can't perfectly order them. Real embeddings have a
  power-law spectrum that lifts recall, consistent with the TurboQuant paper's **>90% recall@10 at
  2–4 bits** on real DBpedia/OpenAI/GloVe data.
- **GloVe sub-sample.** Ground truth is computed by brute-force cosine within the 100k sub-sample.
  The ann-benchmarks pre-computed indices reference the full 1.18M corpus; using them for a sub-sample
  yields misleadingly low recall@100 (only ~8% of true 1.18M neighbors land in 100k).
- **TQ+ calibration** (`calibrate: true`) is opt-in and off in these runs. It can lift recall on real
  embeddings; neutral-to-negative on well-conditioned synthetic data.
- **FWHT** is used automatically for power-of-two dims (128, 256, 512, 768, 1024, 1536…); O(d·log d)
  vs O(d²) for the dense rotation — ~25× faster encode at no recall cost.

## IVF coarse quantizer (synthetic, clustered)

`npm run bench:ivf` — Gaussian-mixture corpus, cosine, 4-bit, sweeping `nprobe` against the flat
scalar baseline (env knobs: `DIM`, `N`, `NQ`, `CLUSTERS`, `NLIST`). Recall is measured against
exact float32 ground truth, so the flat row is the 4-bit quantizer's own recall ceiling.
`nprobe = nlist` reproduces the flat scan exactly (the `searchSlots` oracle).

**20k vectors** (default: `nlist=128`, 64 clusters):

| config  | recall@10 | QPS  | speedup vs flat |
| ------- | --------- | ---- | --------------- |
| flat    | 0.603     | 53   | 1.0×            |
| ivf@1   | 0.387     | 1205 | 22.8×           |
| ivf@4   | 0.602     | 852  | 16.1×           |
| ivf@8   | 0.603     | 600  | **11.4×**       |
| ivf@128 | 0.603     | 60   | 1.1×            |

**200k vectors** (`N=200000 NQ=100 CLUSTERS=256 NLIST=1024`):

| config   | recall@10 | QPS | speedup vs flat |
| -------- | --------- | --- | --------------- |
| flat     | 0.514     | 5   | 1.0×            |
| ivf@1    | 0.202     | 577 | 111.6×          |
| ivf@4    | 0.461     | 457 | 88.4×           |
| ivf@8    | 0.512     | 335 | **64.8×**       |
| ivf@16   | 0.513     | 233 | 45.0×           |
| ivf@1024 | 0.514     | 5   | 1.0×            |

The speedup grows with corpus size: at nprobe=8, probing 8/1024 cells (0.78%) gives **64.8×** on
200k vectors vs **11.4×** on 20k (the O(n·nprobe/nlist) vs O(n) scaling). Recall ceiling is lower
on 200k because the quantizer's 4-bit approximation is harder on a larger, denser corpus.
