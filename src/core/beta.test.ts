import { describe, expect, it } from 'vitest';
import {
  adaptiveSimpson,
  betaCdf,
  betaPdf,
  betaQuantile,
  coordCdf,
  coordPdf,
  coordQuantile,
  lgamma,
} from './beta';

describe('lgamma', () => {
  it('matches factorials: lgamma(n+1) = log(n!)', () => {
    // 0! = 1, 1! = 1, 2! = 2, 5! = 120, 10! = 3628800
    expect(Math.exp(lgamma(1))).toBeCloseTo(1, 9);
    expect(Math.exp(lgamma(2))).toBeCloseTo(1, 9);
    expect(Math.exp(lgamma(3))).toBeCloseTo(2, 9);
    expect(Math.exp(lgamma(6))).toBeCloseTo(120, 6);
    expect(Math.exp(lgamma(11))).toBeCloseTo(3628800, 2);
  });

  it('matches gamma(1/2) = sqrt(pi)', () => {
    expect(Math.exp(lgamma(0.5))).toBeCloseTo(Math.sqrt(Math.PI), 10);
  });
});

describe('adaptiveSimpson', () => {
  it('integrates sin from 0 to pi to 2', () => {
    const v = adaptiveSimpson((x) => Math.sin(x), 0, Math.PI, 1e-12);
    expect(Math.abs(v - 2)).toBeLessThan(1e-9);
  });

  it('integrates a standard Gaussian over a wide window to ~1', () => {
    const norm = 1 / Math.sqrt(2 * Math.PI);
    const f = (x: number) => norm * Math.exp(-(x * x) / 2);
    const v = adaptiveSimpson(f, -20, 20, 1e-12);
    expect(Math.abs(v - 1)).toBeLessThan(1e-9);
  });

  it('integrates a polynomial exactly', () => {
    // ∫_0^2 (3x^2 + 2x + 1) dx = x^3 + x^2 + x | = 8 + 4 + 2 = 14
    const v = adaptiveSimpson((x) => 3 * x * x + 2 * x + 1, 0, 2, 1e-12);
    expect(Math.abs(v - 14)).toBeLessThan(1e-9);
  });

  it('returns 0 for a degenerate interval', () => {
    expect(adaptiveSimpson((x) => x * x, 3, 3, 1e-9)).toBe(0);
  });
});

describe('betaCdf', () => {
  it('is 0 at x=0 and 1 at x=1', () => {
    expect(betaCdf(0, 2, 3)).toBe(0);
    expect(betaCdf(1, 2, 3)).toBe(1);
    expect(betaCdf(0, 0.5, 0.5)).toBe(0);
    expect(betaCdf(1, 0.5, 0.5)).toBe(1);
  });

  it('is monotone increasing on [0,1]', () => {
    for (const [a, b] of [
      [2, 3],
      [0.5, 0.5],
      [5, 1],
      [10, 10],
    ] as const) {
      let prev = -1;
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const c = betaCdf(x, a, b);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
        expect(c).toBeGreaterThanOrEqual(prev - 1e-12);
        prev = c;
      }
    }
  });

  it('matches the symmetry betaCdf(x,a,a) = 1 - betaCdf(1-x,a,a)', () => {
    for (const a of [0.5, 1, 2, 7.5]) {
      for (const x of [0.1, 0.3, 0.5, 0.8]) {
        expect(betaCdf(x, a, a)).toBeCloseTo(1 - betaCdf(1 - x, a, a), 10);
      }
    }
  });

  it('matches the analytic uniform case Beta(1,1): cdf(x)=x', () => {
    for (const x of [0.1, 0.25, 0.5, 0.7, 0.99]) {
      expect(betaCdf(x, 1, 1)).toBeCloseTo(x, 10);
    }
  });

  it('equals the integral of the pdf (cross-check via adaptiveSimpson)', () => {
    for (const [a, b] of [
      [2, 5],
      [3, 3],
      [0.7, 1.3],
    ] as const) {
      for (const x of [0.2, 0.5, 0.85]) {
        const viaIntegral = adaptiveSimpson((t) => betaPdf(t, a, b), 0, x, 1e-12);
        expect(betaCdf(x, a, b)).toBeCloseTo(viaIntegral, 8);
      }
    }
  });
});

describe('betaQuantile', () => {
  it('inverts betaCdf: betaCdf(betaQuantile(p)) ≈ p', () => {
    for (const [a, b] of [
      [2, 3],
      [0.5, 0.5],
      [5, 2],
      [10, 10],
    ] as const) {
      for (const p of [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) {
        const x = betaQuantile(p, a, b);
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(1);
        expect(betaCdf(x, a, b)).toBeCloseTo(p, 8);
      }
    }
  });

  it('returns boundary values for p=0 and p=1', () => {
    expect(betaQuantile(0, 2, 3)).toBe(0);
    expect(betaQuantile(1, 2, 3)).toBe(1);
  });

  it('median of a symmetric Beta(a,a) is 0.5', () => {
    expect(betaQuantile(0.5, 3, 3)).toBeCloseTo(0.5, 9);
    expect(betaQuantile(0.5, 0.5, 0.5)).toBeCloseTo(0.5, 9);
  });
});

describe('coordinate helpers on [-1, 1]', () => {
  it('coordCdf(0, d) ≈ 0.5 (symmetry about 0)', () => {
    for (const d of [2, 8, 64, 1536]) {
      expect(coordCdf(0, d)).toBeCloseTo(0.5, 9);
    }
  });

  it('coordCdf is monotone on [-1,1] with cdf(-1)=0, cdf(1)=1', () => {
    const d = 16;
    expect(coordCdf(-1, d)).toBeCloseTo(0, 9);
    expect(coordCdf(1, d)).toBeCloseTo(1, 9);
    let prev = -1;
    for (let i = -50; i <= 50; i++) {
      const x = i / 50;
      const c = coordCdf(x, d);
      expect(c).toBeGreaterThanOrEqual(prev - 1e-12);
      prev = c;
    }
  });

  it('coordPdf integrates to 1 over [-1,1]', () => {
    for (const d of [2, 8, 64]) {
      const total = adaptiveSimpson((x) => coordPdf(x, d), -1, 1, 1e-11);
      expect(total).toBeCloseTo(1, 7);
    }
  });

  it('coordinate variance ≈ 1/d (∫ x^2 coordPdf dx)', () => {
    for (const d of [8, 64, 1536]) {
      const variance = adaptiveSimpson((x) => x * x * coordPdf(x, d), -1, 1, 1e-12);
      expect(variance).toBeCloseTo(1 / d, Math.min(6, Math.floor(Math.log10(d)) + 4));
      // Relative tolerance check as well.
      expect(Math.abs(variance - 1 / d) / (1 / d)).toBeLessThan(1e-3);
    }
  });

  it('coordQuantile inverts coordCdf and is antisymmetric', () => {
    for (const d of [2, 8, 64]) {
      for (const p of [0.05, 0.25, 0.5, 0.75, 0.95]) {
        const x = coordQuantile(p, d);
        expect(x).toBeGreaterThanOrEqual(-1);
        expect(x).toBeLessThanOrEqual(1);
        expect(coordCdf(x, d)).toBeCloseTo(p, 7);
      }
      // Antisymmetry: q(p) = -q(1-p).
      expect(coordQuantile(0.3, d)).toBeCloseTo(-coordQuantile(0.7, d), 8);
      expect(coordQuantile(0.5, d)).toBeCloseTo(0, 9);
    }
  });
});

describe('argument validation (typed errors)', () => {
  it('betaCdf rejects out-of-range x and bad params', () => {
    expect(() => betaCdf(-0.1, 2, 3)).toThrowError(expect.objectContaining({ code: 'INVALID_X' }));
    expect(() => betaCdf(1.1, 2, 3)).toThrowError(expect.objectContaining({ code: 'INVALID_X' }));
    expect(() => betaCdf(0.5, 0, 3)).toThrowError(
      expect.objectContaining({ code: 'INVALID_PARAM' }),
    );
    expect(() => betaCdf(0.5, 2, -1)).toThrowError(
      expect.objectContaining({ code: 'INVALID_PARAM' }),
    );
    expect(() => betaCdf(NaN, 2, 3)).toThrowError(expect.objectContaining({ code: 'INVALID_X' }));
  });

  it('betaQuantile rejects p outside [0,1]', () => {
    expect(() => betaQuantile(-0.01, 2, 3)).toThrowError(
      expect.objectContaining({ code: 'INVALID_P' }),
    );
    expect(() => betaQuantile(1.01, 2, 3)).toThrowError(
      expect.objectContaining({ code: 'INVALID_P' }),
    );
  });

  it('coordinate helpers reject d < 2', () => {
    expect(() => coordPdf(0, 1)).toThrowError(expect.objectContaining({ code: 'INVALID_DIM' }));
    expect(() => coordCdf(0, 1.5)).toThrowError(expect.objectContaining({ code: 'INVALID_DIM' }));
    expect(() => coordQuantile(0.5, 0)).toThrowError(
      expect.objectContaining({ code: 'INVALID_DIM' }),
    );
  });

  it('coord helpers reject x outside [-1,1] and p outside [0,1]', () => {
    expect(() => coordPdf(-1.5, 8)).toThrowError(expect.objectContaining({ code: 'INVALID_X' }));
    expect(() => coordCdf(2, 8)).toThrowError(expect.objectContaining({ code: 'INVALID_X' }));
    expect(() => coordQuantile(2, 8)).toThrowError(expect.objectContaining({ code: 'INVALID_P' }));
  });
});
