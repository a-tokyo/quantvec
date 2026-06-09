# Roadmap

quantvec is built in waves with a doer / verifier / devil's-advocate subagent workflow; each wave
merges only when the gate is green (typecheck, lint, tests, coverage) and reviewed. Live status lives
in [`docs/worklog/PROGRESS.md`](https://github.com/a-tokyo/quantvec/blob/main/docs/worklog/PROGRESS.md).

## Shipped

- **Core math** — seeded RNG, orthonormal rotation (Householder QR), Beta pdf/cdf/quantile, Lloyd-Max
  codebooks validated against a scipy reference and the paper's distortion bounds.
- **Encode pipeline** — normalize → rotate → quantize → RaBitQ scale.
- **Flat search** — per-query nibble-LUT scan, three metrics, masking, bounded-heap top-k.
- **Index classes** — `TurboQuantIndex` (positional) and `IdMapIndex` (stable `number`/`string`/`bigint`
  ids), both 100% covered.
- **Persistence** — single versioned `QVEC` format with untrusted-input-hardened loading; Node fs helpers.
- **Bit-packed serialization** — codes stored at true 2/3/4 bits on disk (7.9–15.7× compression, on par
  with native TurboQuant implementations).
- **FWHT rotation** — for power-of-two dims, an exact Randomized Hadamard Transform (O(d·log d) build
  and apply; ~25× faster encode than the dense O(d²)/O(d³) Householder rotation) with equal-or-better
  recall. `createRotation` picks it automatically; non-power-of-two dims keep the dense rotation
  (truncation would degrade recall). A deterministic function of `dim` — no serialization change.
- **TQ+ per-coordinate calibration** — opt-in (`calibrate: true`), auto-fit from the first ≥1000-vector
  batch and frozen. Remaps each rotated coordinate onto the canonical marginal; **data-dependent**
  (helps real embeddings, neutral-to-negative on well-conditioned synthetic data), hence off by default.

- **WASM scoring kernel** — an AssemblyScript kernel (`assembly/index.ts`) with the index's codes
  resident in linear memory (uploaded once per mutation, not per query). It accumulates in f64 over the
  same order as the scalar oracle, so results are **bit-identical** (an exact acceleration, ~1.3×
  faster query), and it is base64-inlined with runtime feature-detection and an automatic pure-TS
  fallback — the kernel is an optimization, never a dependency.

## In progress

- **v128 FastScan** — a SIMD swizzle kernel (blocked-nibble layout + u8 LUT + `v128.swizzle` + u16
  accumulators + a top-k rescore) for a larger query speedup on top of the exact WASM kernel.
- **Ergonomic layer** — qdrant-style `createCollection`, `Point<P>` payloads, and a `must`/`should`/
  `must_not` filter DSL compiled to masks.

## Planned

- **Real-dataset benchmark suite** (GloVe, DBpedia/OpenAI) with recall/QPS/latency/memory tables —
  where TQ+ calibration's recall lift is expected to show.
- **IVF / coarse quantizer** for sublinear search on 10M+ corpora.
- **v0.1.0 release** to npm after a full panel + security review.

## Non-goals (for now)

- An HNSW graph index — quantvec is deliberately a flat quantized index; IVF is the planned path to scale.
- A trained/learned codebook — the data-oblivious, zero-training property is the point.
