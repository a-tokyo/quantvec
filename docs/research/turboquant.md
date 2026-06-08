# TurboQuant — distilled notes

Source: _TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate_ —
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research).
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874). PDF: `turboquant-2504.19874.pdf`.

> These notes are our own distillation written to implement quantvec clean-room. Equations are
> paraphrased from the paper; cite this file (and the paper) from code, not any external code.

## Problem

Quantize high-dimensional vectors with minimal distortion in **two senses**:

- **MSE** — reconstruction error ‖x − Q⁻¹(Q(x))‖² (good for the database side / KV-cache).
- **Inner product** — unbiased estimate of ⟨q, x⟩ (good for the query side / retrieval).

Prior methods (PQ, etc.) need training and don't hit optimal distortion rates. TurboQuant is
**data-oblivious** (no training, no parameter tuning) and provably near-optimal.

## Key insight — random rotation induces a known marginal

Apply a random orthonormal Π (QR of a Gaussian matrix). For a unit vector, Πx is uniform on the
sphere, so each coordinate follows **Beta((d−1)/2, (d−1)/2)** on [−1, 1] (→ N(0, 1/d) for large d),
_independent of the input data_, and coordinates are near-independent in high d. Therefore an
optimal **per-coordinate scalar quantizer** suffices — and it can be precomputed from the
distribution alone (no data needed).

## Q_mse — MSE-optimal quantizer (Algorithm 1)

1. Rotate: y = Πx.
2. Quantize each coordinate to the nearest **Lloyd-Max** centroid for the Beta marginal.
3. Reconstruct: x̃ = Πᵀỹ.

Codebook = 1-D Lloyd-Max (k-means on the Beta density) per bit-width b ∈ {1,2,3,4}: alternate
boundaries = midpoints(centroids), centroids = conditional means over Beta — precompute offline.

**Theorem 1 (MSE upper bound):** D_mse ≤ (√3·π/2)·4^−b ≈ **{0.36, 0.117, 0.030, 0.009}** for b=1..4.

## Q_prod — unbiased inner-product quantizer (Algorithm 2)

MSE quantizers are **biased** for inner products (≈ 2/π bias at b=1). Two-stage fix:

1. Apply Q_mse at **(b−1)** bits; take residual r = x − Q_mse⁻¹(Q_mse(x)).
2. Apply a **1-bit Quantized Johnson–Lindenstrauss (QJL)** transform to r: sign(S·r).
3. Reconstruct: x̃ = Q_mse⁻¹(idx) + (√(π/2)/d)·‖r‖·Sᵀ·qjl.

**Theorem 2:** E[⟨y, x̃⟩] = ⟨y, x⟩ (unbiased); distortion D_prod ≤ (√3·π²·‖y‖²/d)·4^−b.

## Lower bounds & optimality (Theorem 3)

For any randomized quantizer: D_mse ≥ 4^−b and D_prod ≥ (1/d)·4^−b. TurboQuant matches these up to
a small constant (**≈ 2.7×** for MSE; tighter, ~1.45×, at b=1).

## Data-obliviousness

Rotation is data-independent; the Beta marginal is universal (depends only on d). Codebooks are
precomputed from the distribution, not data → applies instantly online (streaming / KV-cache / ingest).

## Experiments (what to mirror in our benchmarks)

- **NN search:** DBpedia/OpenAI embeddings d=1536 & 3072; GloVe d=200. Metric: recall@k. Beats PQ on
  recall with **~zero indexing time** (Table 2: TurboQuant ≈ 0.0007–0.0021s vs PQ 37–494s).
- **KV-cache:** quality-neutral at 3.5 bits/channel, marginal degradation at 2.5 bits/channel.

## What quantvec adopts (see DECISIONS.md / architecture.md)

- Random rotation + Beta Lloyd-Max codebooks (Q_mse core) — exactly as above.
- For inner-product correction we ship **RaBitQ per-vector length renormalization** as v1 (simpler,
  SIMD-friendly); the **QJL two-stage above is an autoresearch experiment** to compare against it.
- Validate our codebooks against the Theorem-1 numbers and an independent scipy reference.
