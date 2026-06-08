# RaBitQ — distilled notes

Source: _RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate
Nearest Neighbor Search_ — [arXiv:2405.12497](https://arxiv.org/abs/2405.12497), SIGMOD 2024.
PDF: `rabitq-2405.12497.pdf`.

> Our own distillation for clean-room implementation. We adopt RaBitQ's per-vector scaling idea as
> quantvec's v1 inner-product correction.

## Core idea

Quantize a D-dim vector to a short bit-string after a **random rotation**, and recover distances via
an **unbiased estimator with a per-vector scaling/correction factor** — unlike PQ, it carries a
**sharp theoretical error bound**, so it doesn't fail pathologically on adversarial data.

## Mechanics

1. **Random rotation** (same family as TurboQuant) decorrelates coordinates / spreads energy.
2. Quantize each rotated, normalized coordinate (1-bit in base RaBitQ; quantvec generalizes to 2–4-bit
   Lloyd-Max levels).
3. Store, per vector, its norm ‖v‖ and a **correction scale** ≈ ‖v‖ / ⟨u, x̂⟩ where u is the rotated
   unit vector and x̂ the reconstructed (de-quantized) direction.
4. **At query time:** rotate the query once, score the packed codes (LUT / bitwise-popcount /
   SIMD), then **multiply by the per-vector scale** → an unbiased estimate of ⟨q, v⟩.

## Why it matters for quantvec

- Gives an **unbiased inner-product** without the QJL residual stage — fewer bits spent, simpler hot loop.
- The scale is a single float per vector; scoring stays a tight typed-array / SIMD loop.
- Pairs cleanly with TurboQuant's rotation + Beta Lloyd-Max codebooks: TurboQuant supplies the
  optimal per-coordinate quantizer, RaBitQ supplies the unbiased query-time estimator.

## Fast scoring

Distances computed with bitwise ops / SIMD over packed codes; quantvec uses a **nibble-split LUT**
scalar kernel (the correctness oracle) and an AssemblyScript **v128** kernel for throughput.

## Reported results

Outperforms PQ and variants on the accuracy–efficiency trade-off, with empirical error matching the
theoretical bound. We reproduce the _recall_ behaviour; raw speed we benchmark against JS-ecosystem
peers and our own kernels (see architecture.md — we do not claim parity with native FAISS).
