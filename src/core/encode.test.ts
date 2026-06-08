import { describe, expect, it } from 'vitest';
import { encodeVector, scoreCodes, createEncodeScratch, EncodeError } from './encode';
import { createDenseRotation } from './rotation';
import { buildCodebook } from './codebook';
import { createRng } from './rng';
import type { Bits } from './codebook';

/** Exact float dot product. */
function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Gaussian-random vector scaled to a target norm (random direction, given length). */
function randomVector(d: number, scale: number, rng: ReturnType<typeof createRng>): Float32Array {
  const v = new Float32Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian() * scale;
  return v;
}

describe('encodeVector — argument validation', () => {
  const rotation = createDenseRotation(8, 1);
  const codebook = buildCodebook(8, 2);

  it('rejects bad dim', () => {
    let err: unknown;
    try {
      encodeVector(new Float32Array(8), { dim: 0, bits: 2, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(EncodeError);
    expect((err as EncodeError).code).toBe('INVALID_DIM');
  });

  it('rejects bad bits', () => {
    let err: unknown;
    try {
      encodeVector(new Float32Array(8), { dim: 8, bits: 5 as 2, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('INVALID_BITS');
  });

  it('rejects vec length != dim', () => {
    let err: unknown;
    try {
      encodeVector(new Float32Array(16), { dim: 8, bits: 2, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('INVALID_LENGTH');
  });

  it('rejects rotation dim mismatch', () => {
    const v = new Float32Array(8).fill(1);
    let err: unknown;
    try {
      encodeVector(v, { dim: 8, bits: 2, rotation: createDenseRotation(16, 1), codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('MISMATCH');
  });

  it('rejects codebook size mismatch (wrong bits)', () => {
    const v = new Float32Array(8).fill(1);
    let err: unknown;
    try {
      // codebook built for 2 bits but bits=3 requested → 8 centroids expected.
      encodeVector(v, { dim: 8, bits: 3, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('MISMATCH');
  });

  it('rejects non-finite input', () => {
    const v = new Float32Array(8).fill(1);
    v[3] = Infinity;
    let err: unknown;
    try {
      encodeVector(v, { dim: 8, bits: 2, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('INVALID_LENGTH');
  });

  it('rejects the zero vector with ZERO_VECTOR', () => {
    let err: unknown;
    try {
      encodeVector(new Float32Array(8), { dim: 8, bits: 2, rotation, codebook });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('ZERO_VECTOR');
  });

  it('rejects scratch of the wrong length', () => {
    const v = new Float32Array(8).fill(1);
    const scratch = createEncodeScratch(16);
    let err: unknown;
    try {
      encodeVector(v, { dim: 8, bits: 2, rotation, codebook, scratch });
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('MISMATCH');
  });
});

describe('encodeVector — basic invariants', () => {
  const d = 64;
  const bits: Bits = 4;
  const rotation = createDenseRotation(d, 7);
  const codebook = buildCodebook(d, bits);

  it('is deterministic for the same input', () => {
    const rng = createRng(5);
    const v = randomVector(d, 2, rng);
    const a = encodeVector(v, { dim: d, bits, rotation, codebook });
    const b = encodeVector(v, { dim: d, bits, rotation, codebook });
    expect(Array.from(a.codes)).toEqual(Array.from(b.codes));
    expect(a.scale).toBe(b.scale);
    expect(a.norm).toBe(b.norm);
  });

  it('stores ‖v‖ in norm and emits codes in [0, 2^bits-1] of length dim', () => {
    const rng = createRng(11);
    const max = (1 << bits) - 1;
    for (let t = 0; t < 20; t++) {
      const v = randomVector(d, 1 + t, rng);
      const enc = encodeVector(v, { dim: d, bits, rotation, codebook });
      expect(Math.abs(enc.norm - Math.sqrt(dot(v, v)))).toBeLessThan(1e-3 * (enc.norm + 1));
      expect(enc.codes.length).toBe(d);
      for (const c of enc.codes) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(max);
      }
      expect(Number.isFinite(enc.scale)).toBe(true);
    }
  });

  it('reuses provided scratch buffers (same result as without)', () => {
    const rng = createRng(13);
    const v = randomVector(d, 3, rng);
    const scratch = createEncodeScratch(d);
    const a = encodeVector(v, { dim: d, bits, rotation, codebook, scratch });
    const b = encodeVector(v, { dim: d, bits, rotation, codebook });
    expect(Array.from(a.codes)).toEqual(Array.from(b.codes));
    expect(a.scale).toBeCloseTo(b.scale, 10);
  });
});

describe('scoreCodes', () => {
  it('computes Σ centroids[codes[i]]·qRot[i]', () => {
    const centroids = new Float32Array([-0.5, -0.1, 0.1, 0.5]);
    const codes = new Uint8Array([0, 3, 1]);
    const qRot = new Float32Array([2, 4, 6]);
    // -0.5*2 + 0.5*4 + -0.1*6 = -1 + 2 - 0.6 = 0.4
    expect(scoreCodes(codes, centroids, qRot)).toBeCloseTo(0.4, 6);
  });

  it('throws MISMATCH on length mismatch', () => {
    let err: unknown;
    try {
      scoreCodes(new Uint8Array(3), new Float32Array(4), new Float32Array(2));
    } catch (e) {
      err = e;
    }
    expect((err as EncodeError).code).toBe('MISMATCH');
  });
});

describe('encodeVector — self-score recovers ‖v‖²', () => {
  // With q = v: scale·scoreCodes(codes, centroids, Q·v) ≈ ⟨v, v⟩ = ‖v‖².
  for (const d of [64, 128]) {
    for (const bits of [2, 3, 4] as const) {
      it(`self dot ≈ ‖v‖² (d=${d}, bits=${bits})`, () => {
        const rotation = createDenseRotation(d, 100 + d);
        const codebook = buildCodebook(d, bits);
        const rng = createRng(7 * d + bits);
        const qRot = new Float32Array(d);
        let worstRel = 0;
        for (let t = 0; t < 30; t++) {
          const v = randomVector(d, 1 + (t % 5), rng);
          const enc = encodeVector(v, { dim: d, bits, rotation, codebook });
          rotation.apply(v, qRot); // q_rot = Q·v
          const est = enc.scale * scoreCodes(enc.codes, codebook.centroids, qRot);
          const truth = dot(v, v);
          worstRel = Math.max(worstRel, Math.abs(est - truth) / truth);
        }
        // Quantization tolerance: loosen as bits shrink. Self-score is the
        // best case (query == database direction), so it is tight.
        const tol = bits === 2 ? 0.1 : bits === 3 ? 0.05 : 0.02;
        expect(worstRel).toBeLessThan(tol);
      });
    }
  }
});

describe('encodeVector — UNBIASEDNESS (key)', () => {
  // Fix one database vector v (encode once). Draw many random queries q and
  // compare the estimator scale·⟨Q·q, c⟩ to the true ⟨q, v⟩. The mean signed
  // error must be ≈ 0 (approximately unbiased), and the RMS error must be small
  // relative to ‖v‖ and shrink as bits increase (4-bit < 3-bit < 2-bit).
  for (const d of [64, 128]) {
    it(`mean error ≈ 0 and RMS shrinks with bits (d=${d})`, () => {
      const rotation = createDenseRotation(d, 555 + d);
      const dbRng = createRng(20 * d);
      const v = randomVector(d, 3, dbRng);
      const vNorm = Math.sqrt(dot(v, v));

      const NUM_QUERIES = 2000;
      const rms: Record<number, number> = {};
      const meanErr: Record<number, number> = {};

      for (const bits of [2, 3, 4] as const) {
        const codebook = buildCodebook(d, bits);
        const enc = encodeVector(v, { dim: d, bits, rotation, codebook });
        // Same query stream per bit-width so the comparison is apples-to-apples.
        const qRng = createRng(999);
        const qRot = new Float32Array(d);
        let sumErr = 0;
        let sumSqErr = 0;
        for (let i = 0; i < NUM_QUERIES; i++) {
          const q = randomVector(d, 1, qRng);
          rotation.apply(q, qRot);
          const est = enc.scale * scoreCodes(enc.codes, codebook.centroids, qRot);
          const truth = dot(q, v);
          const e = est - truth;
          sumErr += e;
          sumSqErr += e * e;
        }
        meanErr[bits] = sumErr / NUM_QUERIES;
        rms[bits] = Math.sqrt(sumSqErr / NUM_QUERIES);
      }

      // Approximately unbiased: mean signed error is small relative to ‖v‖.
      // (Each query has E[⟨q,v⟩]=0 with std ≈ ‖v‖/√d·‖q‖, so the empirical mean
      // over 2000 queries has its own sampling noise; the bound below is the
      // bias+noise budget.)
      for (const bits of [2, 3, 4] as const) {
        expect(Math.abs(meanErr[bits]!)).toBeLessThan(0.05 * vNorm);
      }

      // RMS is small relative to ‖v‖ and strictly decreases with more bits.
      // Measured RMS/‖v‖ ≈ {0.36, 0.18, 0.08} (d=64) and {0.38, 0.19, 0.09}
      // (d=128) — roughly halving per added bit, matching ~6 dB/bit Lloyd-Max.
      expect(rms[2]!).toBeLessThan(0.5 * vNorm);
      expect(rms[3]!).toBeLessThan(rms[2]!);
      expect(rms[4]!).toBeLessThan(rms[3]!);
    });
  }
});

describe('encodeVector — recall sanity (top-1)', () => {
  // Small dataset: the estimated-score top-1 should match (or be within top-k of)
  // the exact-dot top-1 for the large majority of queries at 4-bit.
  it('top-1 by estimate is within exact top-5 most of the time (n=200, d=64, 4-bit)', () => {
    const d = 64;
    const bits: Bits = 4;
    const n = 200;
    const rotation = createDenseRotation(d, 314);
    const codebook = buildCodebook(d, bits);

    // Build a database of n vectors; encode each.
    const dbRng = createRng(2718);
    const db: Float32Array[] = [];
    const encs: { codes: Uint8Array; scale: number }[] = [];
    for (let i = 0; i < n; i++) {
      const v = randomVector(d, 1 + (i % 7) * 0.3, dbRng);
      db.push(v);
      const e = encodeVector(v, { dim: d, bits, rotation, codebook });
      encs.push({ codes: e.codes, scale: e.scale });
    }

    const qRng = createRng(161803);
    const qRot = new Float32Array(d);
    const NUM_QUERIES = 60;
    const TOPK = 5;
    let hits = 0;
    for (let qi = 0; qi < NUM_QUERIES; qi++) {
      const q = randomVector(d, 1, qRng);
      rotation.apply(q, qRot);

      // Exact ranking by true dot product.
      const exact = db.map((v, i) => ({ i, s: dot(q, v) }));
      exact.sort((a, b) => b.s - a.s);
      const exactTopK = new Set(exact.slice(0, TOPK).map((e) => e.i));

      // Estimated top-1.
      let bestI = 0;
      let bestEst = -Infinity;
      for (let i = 0; i < n; i++) {
        const est = encs[i]!.scale * scoreCodes(encs[i]!.codes, codebook.centroids, qRot);
        if (est > bestEst) {
          bestEst = est;
          bestI = i;
        }
      }
      if (exactTopK.has(bestI)) hits++;
    }
    // At 4-bit the estimated top-1 should fall inside the exact top-5 for the
    // large majority of queries.
    expect(hits / NUM_QUERIES).toBeGreaterThan(0.8);
  });
});
