import { describe, expect, it } from 'vitest';
import { buildCodebook, quantizeCoord, mseDistortion, CodebookError } from './codebook';

describe('buildCodebook — argument validation', () => {
  it('rejects bits outside {2,3,4} with CodebookError(INVALID_BITS)', () => {
    for (const bad of [0, 1, 5, 8, 2.5, NaN]) {
      let err: unknown;
      try {
        // Cast through unknown to test the runtime guard with out-of-type values.
        buildCodebook(8, bad as 2);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(CodebookError);
      expect((err as CodebookError).code).toBe('INVALID_BITS');
    }
  });

  it('rejects invalid dim with CodebookError(INVALID_DIM)', () => {
    for (const bad of [0, 1, 7, 9, 12, -8, 8.5, NaN, Infinity]) {
      let err: unknown;
      try {
        buildCodebook(bad, 2);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(CodebookError);
      expect((err as CodebookError).code).toBe('INVALID_DIM');
    }
  });

  it('produces 2^bits centroids and 2^bits+1 boundaries', () => {
    for (const bits of [2, 3, 4] as const) {
      const n = 1 << bits;
      const cb = buildCodebook(64, bits);
      expect(cb.centroids.length).toBe(n);
      expect(cb.boundaries.length).toBe(n + 1);
      expect(cb.centroids).toBeInstanceOf(Float32Array);
      expect(cb.boundaries).toBeInstanceOf(Float32Array);
    }
  });
});

describe('buildCodebook — structural invariants', () => {
  for (const dim of [8, 16, 64]) {
    for (const bits of [2, 3, 4] as const) {
      it(`centroids strictly increasing, symmetric, interleaved (dim=${dim}, bits=${bits})`, () => {
        const n = 1 << bits;
        const { boundaries, centroids } = buildCodebook(dim, bits);

        // Centroids strictly increasing.
        for (let i = 1; i < n; i++) {
          expect(centroids[i]!).toBeGreaterThan(centroids[i - 1]!);
        }

        // Symmetric about 0: c_i ≈ -c_{n-1-i} (the density f is symmetric).
        for (let i = 0; i < n; i++) {
          expect(Math.abs(centroids[i]! + centroids[n - 1 - i]!)).toBeLessThan(1e-3);
        }

        // Boundaries: endpoints -1 and 1, strictly increasing, interleave centroids.
        expect(boundaries[0]).toBe(-1);
        expect(boundaries[n]).toBe(1);
        for (let i = 1; i <= n; i++) {
          expect(boundaries[i]!).toBeGreaterThan(boundaries[i - 1]!);
        }
        // Each centroid lies strictly inside its own cell.
        for (let i = 0; i < n; i++) {
          expect(centroids[i]!).toBeGreaterThan(boundaries[i]!);
          expect(centroids[i]!).toBeLessThan(boundaries[i + 1]!);
        }
      });
    }
  }
});

describe('buildCodebook — determinism (no RNG)', () => {
  it('identical output for identical (dim, bits)', () => {
    const a = buildCodebook(64, 3);
    const b = buildCodebook(64, 3);
    expect(Array.from(a.centroids)).toEqual(Array.from(b.centroids));
    expect(Array.from(a.boundaries)).toEqual(Array.from(b.boundaries));
  });
});

describe('mseDistortion — TurboQuant Theorem 1 envelope', () => {
  // Theorem 1: per-coordinate MSE of Q_mse ≲ (√3·π/2)·4^−b, with the coordinate
  // density variance scaled by 1/d (each coordinate carries 1/d of the unit
  // energy). The bound numbers are {0.117, 0.030, 0.009} for b=2,3,4 BEFORE the
  // 1/d scaling. We assert the measured per-coordinate distortion is below the
  // (generously scaled) envelope and decreases by roughly 4× per added bit.
  const ENVELOPE = (b: number) => (Math.sqrt(3) * Math.PI) / 2 / Math.pow(4, b);

  for (const dim of [8, 16, 64]) {
    it(`distortion below Theorem-1 envelope & ~4× decrease per bit (dim=${dim})`, () => {
      const dists: number[] = [];
      for (const bits of [2, 3, 4] as const) {
        const cb = buildCodebook(dim, bits);
        const d = mseDistortion(dim, bits, cb);
        // Variance of one coordinate is 1/d ≤ 1/8, so the distortion (which is a
        // fraction of that variance) is comfortably below the unscaled envelope.
        expect(d).toBeLessThan(ENVELOPE(bits));
        expect(d).toBeGreaterThan(0);
        dists.push(d);
      }
      // Monotonically decreasing in b.
      expect(dists[1]!).toBeLessThan(dists[0]!);
      expect(dists[2]!).toBeLessThan(dists[1]!);
      // Roughly 4× per added bit (Lloyd-Max scalar quantizer ≈ 6 dB/bit). Allow a
      // wide band: ratio in [2.5, 5].
      const r1 = dists[0]! / dists[1]!;
      const r2 = dists[1]! / dists[2]!;
      for (const r of [r1, r2]) {
        expect(r).toBeGreaterThan(2.5);
        expect(r).toBeLessThan(5);
      }
    });
  }
});

describe('quantizeCoord — bucketing', () => {
  it('maps sample points to the nearest-centroid cell', () => {
    const dim = 16;
    const bits = 3 as const;
    const n = 1 << bits;
    const { boundaries, centroids } = buildCodebook(dim, bits);

    // For a grid of x values, the bucket index must (a) be in [0, n-1] and
    // (b) have x fall within [boundaries[idx], boundaries[idx+1]], i.e. the
    // nearest-centroid cell by Lloyd-Max boundaries (which are centroid midpoints).
    for (let t = 0; t <= 200; t++) {
      const x = -1 + (2 * t) / 200;
      const idx = quantizeCoord(x, boundaries);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(n);
      expect(x).toBeGreaterThanOrEqual(boundaries[idx]!);
      expect(x).toBeLessThanOrEqual(boundaries[idx + 1]!);
      // The chosen centroid must be the nearest of all centroids.
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < n; i++) {
        const dd = Math.abs(x - centroids[i]!);
        if (dd < bestDist) {
          bestDist = dd;
          best = i;
        }
      }
      expect(idx).toBe(best);
    }
  });

  it('clamps out-of-range inputs into the extreme cells', () => {
    const { boundaries } = buildCodebook(8, 2);
    const n = boundaries.length - 1;
    expect(quantizeCoord(-5, boundaries)).toBe(0);
    expect(quantizeCoord(5, boundaries)).toBe(n - 1);
    expect(quantizeCoord(-1, boundaries)).toBe(0);
    expect(quantizeCoord(1, boundaries)).toBe(n - 1);
  });
});

describe('buildCodebook — scipy cross-check (validation oracle)', () => {
  // Independent Lloyd-Max reference computed with scipy 1.13.1
  // (scipy.stats.beta + scipy.integrate.quad). Generated offline and embedded as
  // constants so the test runs node-free. Regenerate with the script in the
  // commit message / local scratch if the algorithm changes.
  //
  //   from scipy.stats import beta
  //   from scipy.integrate import quad
  //   coord_pdf(x,d) = 0.5*beta.pdf((x+1)/2,(d-1)/2,(d-1)/2)
  //   Lloyd-Max: boundaries=midpoints(centroids) [-1..1], centroid=∫x f/∫f.
  const REF: Array<{
    dim: number;
    bits: 2 | 3 | 4;
    centroids: number[];
    boundaries: number[];
  }> = [
    {
      dim: 8,
      bits: 2,
      centroids: [-0.5048246226, -0.1579220992, 0.1579220992, 0.5048246226],
      boundaries: [-1.0, -0.3313733609, 0.0, 0.3313733609, 1.0],
    },
    {
      dim: 64,
      bits: 3,
      centroids: [
        -0.2639139308, -0.1661678589, -0.0938322632, -0.0304691789, 0.0304691789, 0.0938322632,
        0.1661678589, 0.2639139308,
      ],
      boundaries: [
        -1.0, -0.2150408949, -0.1300000611, -0.0621507211, 0.0, 0.0621507211, 0.1300000611,
        0.2150408949, 1.0,
      ],
    },
  ];

  for (const { dim, bits, centroids, boundaries } of REF) {
    it(`matches scipy within 1e-3 (dim=${dim}, bits=${bits})`, () => {
      const cb = buildCodebook(dim, bits);
      expect(cb.centroids.length).toBe(centroids.length);
      expect(cb.boundaries.length).toBe(boundaries.length);
      for (let i = 0; i < centroids.length; i++) {
        expect(Math.abs(cb.centroids[i]! - centroids[i]!)).toBeLessThan(1e-3);
      }
      for (let i = 0; i < boundaries.length; i++) {
        expect(Math.abs(cb.boundaries[i]! - boundaries[i]!)).toBeLessThan(1e-3);
      }
    });
  }
});
