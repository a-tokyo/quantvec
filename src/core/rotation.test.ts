import { describe, expect, it } from 'vitest';
import {
  createDenseRotation,
  createHadamardRotation,
  createRotation,
  RotationError,
} from './rotation';
import { createRng } from './rng';

/** Dot product of two equal-length Float32Arrays. */
function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Euclidean norm. */
function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

/** A random unit vector in R^d from the given RNG (Gaussian → normalize). */
function randomUnitVector(d: number, rng: ReturnType<typeof createRng>): Float32Array {
  const v = new Float32Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  const n = norm(v);
  for (let i = 0; i < d; i++) v[i] = v[i]! / n;
  return v;
}

describe('createDenseRotation — argument validation', () => {
  it('rejects non-multiples of 8 with RotationError(INVALID_DIM)', () => {
    for (const bad of [0, 1, 7, 9, 12, 100, -8]) {
      let err: unknown;
      try {
        createDenseRotation(bad);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RotationError);
      expect((err as RotationError).code).toBe('INVALID_DIM');
    }
  });

  it('rejects non-integer / non-finite dims', () => {
    for (const bad of [8.5, NaN, Infinity]) {
      let err: unknown;
      try {
        createDenseRotation(bad);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RotationError);
      expect((err as RotationError).code).toBe('INVALID_DIM');
    }
  });

  it('accepts positive multiples of 8', () => {
    for (const d of [8, 16, 64]) {
      const r = createDenseRotation(d);
      expect(r.dim).toBe(d);
    }
  });
});

describe('createDenseRotation — orthonormality (QᵀQ ≈ I)', () => {
  for (const d of [8, 16, 64]) {
    it(`Q is orthonormal for d=${d}`, () => {
      const r = createDenseRotation(d, 12345);
      // Compute Q via applying to the standard basis: column j of the matrix
      // mapping is apply(e_j). We then check that the columns are orthonormal,
      // which is QᵀQ = I (the rotation's apply is x ↦ Qx).
      const cols: Float32Array[] = [];
      for (let j = 0; j < d; j++) {
        const e = new Float32Array(d);
        e[j] = 1;
        const out = new Float32Array(d);
        r.apply(e, out);
        cols.push(out);
      }
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          const g = dot(cols[i]!, cols[j]!);
          const expected = i === j ? 1 : 0;
          expect(Math.abs(g - expected)).toBeLessThan(1e-4);
        }
      }
    });
  }
});

describe('createDenseRotation — determinism', () => {
  it('same seed ⇒ identical mapping', () => {
    const d = 16;
    const a = createDenseRotation(d, 777);
    const b = createDenseRotation(d, 777);
    const rng = createRng(1);
    for (let t = 0; t < 5; t++) {
      const x = randomUnitVector(d, rng);
      const ya = new Float32Array(d);
      const yb = new Float32Array(d);
      a.apply(x, ya);
      b.apply(x, yb);
      for (let i = 0; i < d; i++) expect(ya[i]).toBe(yb[i]);
    }
  });

  it('different seeds ⇒ different mapping', () => {
    const d = 16;
    const a = createDenseRotation(d, 1);
    const b = createDenseRotation(d, 2);
    const x = new Float32Array(d);
    x[0] = 1;
    const ya = new Float32Array(d);
    const yb = new Float32Array(d);
    a.apply(x, ya);
    b.apply(x, yb);
    let differs = false;
    for (let i = 0; i < d; i++) if (Math.abs(ya[i]! - yb[i]!) > 1e-6) differs = true;
    expect(differs).toBe(true);
  });

  it('default seed is deterministic', () => {
    const d = 8;
    const a = createDenseRotation(d);
    const b = createDenseRotation(d);
    const x = new Float32Array(d).fill(0.5);
    const ya = new Float32Array(d);
    const yb = new Float32Array(d);
    a.apply(x, ya);
    b.apply(x, yb);
    for (let i = 0; i < d; i++) expect(ya[i]).toBe(yb[i]);
  });
});

describe('createDenseRotation — norm preservation & inverse', () => {
  for (const d of [8, 16, 64]) {
    it(`‖apply(x)‖ ≈ ‖x‖ for d=${d}`, () => {
      const r = createDenseRotation(d, 42);
      const rng = createRng(99);
      for (let t = 0; t < 10; t++) {
        const x = new Float32Array(d);
        for (let i = 0; i < d; i++) x[i] = rng.nextGaussian() * 3;
        const y = new Float32Array(d);
        r.apply(x, y);
        expect(Math.abs(norm(y) - norm(x))).toBeLessThan(1e-3 * (norm(x) + 1));
      }
    });

    it(`applyTranspose(apply(x)) ≈ x for d=${d}`, () => {
      const r = createDenseRotation(d, 7);
      const rng = createRng(123);
      for (let t = 0; t < 10; t++) {
        const x = new Float32Array(d);
        for (let i = 0; i < d; i++) x[i] = rng.nextGaussian();
        const y = new Float32Array(d);
        const back = new Float32Array(d);
        r.apply(x, y);
        r.applyTranspose(y, back);
        for (let i = 0; i < d; i++) expect(Math.abs(back[i]! - x[i]!)).toBeLessThan(1e-4);
      }
    });
  }
});

describe('createDenseRotation — apply/applyTranspose argument validation', () => {
  it('rejects mismatched src/dst lengths', () => {
    const d = 8;
    const r = createDenseRotation(d, 1);
    const good = new Float32Array(d);
    const bad = new Float32Array(d + 8);
    for (const fn of ['apply', 'applyTranspose'] as const) {
      let e1: unknown;
      try {
        r[fn](bad, good);
      } catch (e) {
        e1 = e;
      }
      expect((e1 as RotationError).code).toBe('INVALID_LENGTH');

      let e2: unknown;
      try {
        r[fn](good, bad);
      } catch (e) {
        e2 = e;
      }
      expect((e2 as RotationError).code).toBe('INVALID_LENGTH');
    }
  });
});

describe('createDenseRotation — Beta-marginal property', () => {
  // TurboQuant's key property: rotating a random unit vector by a fixed
  // orthonormal Q makes each coordinate ~ Beta((d-1)/2,(d-1)/2) on [-1,1],
  // whose variance is 1/d. We verify a coordinate's sample variance ≈ 1/d.
  for (const d of [16, 64]) {
    it(`coordinate sample variance ≈ 1/d for d=${d}`, () => {
      const r = createDenseRotation(d, 2024);
      const rng = createRng(31337);
      const samples = 4000;
      // Track the first coordinate across many random unit inputs.
      let sum = 0;
      let sumSq = 0;
      const y = new Float32Array(d);
      for (let s = 0; s < samples; s++) {
        const x = randomUnitVector(d, rng);
        r.apply(x, y);
        const c = y[0]!;
        sum += c;
        sumSq += c * c;
      }
      const mean = sum / samples;
      const variance = sumSq / samples - mean * mean;
      const expected = 1 / d;
      // Mean ≈ 0 (symmetric density).
      expect(Math.abs(mean)).toBeLessThan(0.05);
      // Sample variance within ~15% of 1/d (Monte-Carlo tolerance).
      expect(Math.abs(variance - expected)).toBeLessThan(0.15 * expected);
    });
  }
});

describe('createDenseRotation — Haar-ness across seeds (fixed canonical input)', () => {
  // The test above feeds an already-uniform random unit vector and so checks the
  // *input* sphere, not Q itself: any orthonormal Q would pass. To genuinely
  // exercise that Q is Haar-distributed (uniform on O(d)), feed a FIXED canonical
  // input e0 = [1,0,…] through Q built from MANY different seeds. For a Haar Q,
  // Q·e0 is its first column — a uniformly random unit vector — so its first
  // coordinate (Q[0,0]) has mean 0 and variance 1/d. A biased / non-uniform
  // construction would skew this sample variance away from 1/d.
  for (const d of [16, 64]) {
    it(`Q·e0 first coordinate has sample variance ≈ 1/d for d=${d}`, { timeout: 30_000 }, () => {
      const e0 = new Float32Array(d);
      e0[0] = 1;
      const y = new Float32Array(d);
      // Each sample builds a fresh O(d³) QR; 2000 seeds is ample for a 1/d
      // variance estimate at the 15% tolerance below (SE ≈ √(2/samples)·(1/d)).
      const samples = 2000;
      let sum = 0;
      let sumSq = 0;
      let unitErr = 0;
      for (let seed = 0; seed < samples; seed++) {
        const r = createDenseRotation(d, seed);
        r.apply(e0, y);
        const c = y[0]!;
        sum += c;
        sumSq += c * c;
        // Q·e0 must itself be a unit vector (Q orthonormal).
        unitErr = Math.max(unitErr, Math.abs(norm(y) - 1));
      }
      const mean = sum / samples;
      const variance = sumSq / samples - mean * mean;
      const expected = 1 / d;
      expect(unitErr).toBeLessThan(1e-3);
      expect(Math.abs(mean)).toBeLessThan(0.05);
      expect(Math.abs(variance - expected)).toBeLessThan(0.15 * expected);
    });
  }
});

describe('createHadamardRotation', () => {
  it('requires a power-of-two dim', () => {
    for (const bad of [0, 7, 24, 768, 1536]) {
      let err: unknown;
      try {
        createHadamardRotation(bad);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RotationError);
      expect((err as RotationError).code).toBe('INVALID_DIM');
    }
  });

  it('preserves the Euclidean norm (orthonormal)', () => {
    const rng = createRng(5);
    const rot = createHadamardRotation(64, 3);
    const y = new Float32Array(64);
    for (let t = 0; t < 5; t++) {
      rot.apply(randomUnitVector(64, rng), y);
      expect(norm(y)).toBeCloseTo(1, 5);
    }
  });

  it('applyTranspose is the exact inverse of apply', () => {
    const rot = createHadamardRotation(128, 2);
    const x = randomUnitVector(128, createRng(8));
    const y = new Float32Array(128);
    const back = new Float32Array(128);
    rot.apply(x, y);
    rot.applyTranspose(y, back);
    for (let i = 0; i < 128; i++) expect(back[i]).toBeCloseTo(x[i]!, 5);
  });

  it('is deterministic in (dim, seed)', () => {
    const x = randomUnitVector(32, createRng(1));
    const ya = new Float32Array(32);
    const yb = new Float32Array(32);
    createHadamardRotation(32, 7).apply(x, ya);
    createHadamardRotation(32, 7).apply(x, yb);
    expect(Array.from(ya)).toEqual(Array.from(yb));
  });

  it('validates src/dst length', () => {
    let err: unknown;
    try {
      createHadamardRotation(16, 0).apply(new Float32Array(8), new Float32Array(16));
    } catch (e) {
      err = e;
    }
    expect((err as RotationError).code).toBe('INVALID_LENGTH');
  });
});

describe('createRotation — dispatch', () => {
  it('uses the Hadamard rotation for a power-of-two dim', () => {
    const x = randomUnitVector(64, createRng(2));
    const a = new Float32Array(64);
    const b = new Float32Array(64);
    createRotation(64, 4).apply(x, a);
    createHadamardRotation(64, 4).apply(x, b);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('uses the dense rotation for a non-power-of-two dim', () => {
    const x = randomUnitVector(24, createRng(2));
    const a = new Float32Array(24);
    const b = new Float32Array(24);
    createRotation(24, 4).apply(x, a);
    createDenseRotation(24, 4).apply(x, b);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});
