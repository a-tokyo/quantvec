# quantvec — architecture & design

How quantvec turns the [TurboQuant](./turboquant.md) + [RaBitQ](./rabitq.md) notes into a concrete,
isomorphic TypeScript library. Clean-room: derived from the papers, not from any existing code.

## Encode pipeline (per vector)

```
raw v ──► normalize (store ‖v‖)
       ──► rotate by Π        (deterministic seeded orthonormal matrix, Householder QR of Gaussian)
       ──► [optional] per-coordinate calibration (map empirical pct → Beta pct)
       ──► Lloyd-Max quantize  (MSE-optimal codebook for Beta((d-1)/2,(d-1)/2), b∈{2,3,4})
       ──► bit-pack            (2/3/4 bits per coord, blocked for fast scoring)
       ──► store RaBitQ scale  (per-vector length renormalization → unbiased ⟨q,v⟩)
```

State: `packed: Uint8Array`, `scales: Float32Array`, `norms`, frozen `rotation`/`codebook`,
optional `calibration`.

## Search path (per query)

`rotate query once → build nibble-split LUT → scan packed codes (scalar kernel = oracle, or WASM
v128) → multiply by per-vector scale → bounded top-k min-heap → optional mask/allowlist filter`.

It is a **flat quantized index**: an O(n) SIMD linear scan over tiny codes (not HNSW). Fast because
codes are 2–4 bits; honest target ~1–10M vectors. IVF/coarse-quantizer is future work.

## Module responsibilities (`src/`)

| Module                   | Responsibility                                            | Named algorithm  |
| ------------------------ | --------------------------------------------------------- | ---------------- |
| `core/rng`               | deterministic seeded RNG                                  | xoshiro/PCG      |
| `core/rotation`          | orthonormal matrix from a seeded Gaussian                 | Householder QR   |
| `core/beta`              | Beta pdf/cdf/quantile + numerical integration             | adaptive Simpson |
| `core/codebook`          | per-(dim,bits) boundaries + centroids                     | Lloyd-Max        |
| `core/encode`            | the pipeline above                                        | —                |
| `core/pack`              | bit-plane ↔ blocked layout                                | —                |
| `core/search`            | LUT scoring + top-k + filter (scalar oracle)              | nibble-split LUT |
| `core/topk`              | bounded best-k                                            | min-heap         |
| `core/metrics`           | cosine / dot / euclid (via stored norms)                  | —                |
| `index/turboquant-index` | positional index, add/search/swapRemove/serialize         | —                |
| `index/id-map-index`     | stable ids (number default; bigint/string opt-in)         | —                |
| `io/serialize`           | versioned ArrayBuffer; bounds-validated `fromBytes`       | —                |
| `wasm/*`                 | AssemblyScript v128 kernel + feature-detect + TS fallback | —                |
| `ergonomic/*`            | Collection, Point, filter DSL → mask                      | —                |

## Distance metrics

Inner product / cosine are primary (cosine = dot after normalization, which we already do).
Euclidean derived from stored norms: ‖a−b‖² = ‖a‖² + ‖b‖² − 2⟨a,b⟩. Manhattan is out of scope for
the quantized estimator (exact-only).

## Serialization format (versioned, validated)

`magic | version | flags | dim(u32) | n(u32) | bits(u8) | packed | scales(f32) | calibration?`.
`fromBytes` treats input as **untrusted**: validate magic/version and every length against the
buffer size _before_ allocating (prevents OOM/OOB from crafted files). `toBytes`/`fromBytes` are
runtime-agnostic (ArrayBuffer); `quantvec/node` adds `node:fs/promises` convenience wrappers.

## IDs

`IdMapIndex` keeps a `Map<id, slot>` + `slot→id` array. Default id type **number** (safe ≤ 2^53);
**bigint/string** opt-in to avoid precision loss without paying bigint cost by default.

## Validation strategy (clean-room oracle)

1. **Codebook**: assert quantvec's Lloyd-Max centroids match an independent **scipy** reference and
   that measured D_mse ≤ (√3π/2)·4^−b (TurboQuant Theorem 1).
2. **Search**: self-query returns self across BLOCK(=32) boundaries (n = 32/33/63/64/65); recall vs
   exact brute-force float32; target recall@10 > 90% at 2–4 bits.
3. **WASM ≡ scalar**: v128 kernel scores equal the scalar oracle within fp tolerance; fallback path
   exercised when SIMD is unavailable.

## WASM-SIMD loading (isomorphic)

`asc` compiles `assembly/` → `build/quantvec.wasm`; a prebuild step base64-inlines it into a
generated TS module so the published bundle needs no separate `.wasm` fetch (works in Node, browser,
Workers). Runtime feature-detects SIMD and falls back to the pure-TS scalar kernel; wasm memory is
explicitly freed (no leaks across queries).
