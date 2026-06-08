import { describe, expect, it } from 'vitest';
import { adaptiveSimpson, IntegrationError } from './integrate';
import { coordPdf } from './beta';

describe('adaptiveSimpson — smooth integrands', () => {
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

  it('integrates a reversed interval (b < a) with the expected sign', () => {
    // ∫_pi^0 sin = -2.
    const v = adaptiveSimpson((x) => Math.sin(x), Math.PI, 0, 1e-12);
    expect(Math.abs(v + 2)).toBeLessThan(1e-9);
  });
});

describe('adaptiveSimpson — termination & fail-fast (CRITICAL 1)', () => {
  it('does NOT hang on a one-sided singular cell; throws a typed error fast', () => {
    // coordPdf(x, 2) = 1/(π√(1-x²)) → ∞ at x=1. The old "nudge" design recursed
    // to max depth forever here. The sound design must reject it immediately.
    const start = Date.now();
    let err: unknown;
    try {
      adaptiveSimpson((x) => coordPdf(x, 2), 0.9, 1.0, 1e-12);
    } catch (e) {
      err = e;
    }
    const elapsed = Date.now() - start;
    expect(err).toBeInstanceOf(IntegrationError);
    expect((err as IntegrationError).code).toBe('NON_FINITE_INTEGRAND');
    // "Fast" — well under a second even on a slow CI box.
    expect(elapsed).toBeLessThan(1000);
  });

  it('throws NON_FINITE_INTEGRAND when the integrand is NaN/Inf anywhere', () => {
    let err: unknown;
    try {
      adaptiveSimpson((x) => (x > 0.5 ? Infinity : 1), 0, 1, 1e-9);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(IntegrationError);
    expect((err as IntegrationError).code).toBe('NON_FINITE_INTEGRAND');

    let err2: unknown;
    try {
      adaptiveSimpson(() => NaN, 0, 1, 1e-9);
    } catch (e) {
      err2 = e;
    }
    expect((err2 as IntegrationError).code).toBe('NON_FINITE_INTEGRAND');
  });

  it('IntegrationError carries name and message', () => {
    const e = new IntegrationError('NON_FINITE_INTEGRAND', 'boom');
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe('IntegrationError');
    expect(e.message).toBe('boom');
    expect(e.code).toBe('NON_FINITE_INTEGRAND');
  });
});

describe('adaptiveSimpson — spiky large-d densities & one-sided sub-cells', () => {
  it('resolves one-sided sub-cells of a large-d (d=1536) density that sum to 1', () => {
    const d = 1536;
    // Split the right half [0,1] at the spike shoulder 0.05; combine with the
    // symmetric left half [-1,0]. The total probability mass must be 1.
    const inner = adaptiveSimpson((x) => coordPdf(x, d), 0.0, 0.05, 1e-13);
    const outer = adaptiveSimpson((x) => coordPdf(x, d), 0.05, 1.0, 1e-13);
    const leftHalf = adaptiveSimpson((x) => coordPdf(x, d), -1.0, 0.0, 1e-13);
    expect(Math.abs(inner + outer + leftHalf - 1)).toBeLessThan(1e-8);
    // The spike carries almost all the mass of the half-cell.
    expect(inner).toBeGreaterThan(outer);
  });

  it('integrates the full large-d density over [-1,1] to 1 (spike not missed)', () => {
    for (const d of [64, 768, 1536]) {
      const total = adaptiveSimpson((x) => coordPdf(x, d), -1, 1, 1e-12);
      expect(Math.abs(total - 1)).toBeLessThan(1e-8);
    }
  });

  it('computes a conditional-mean numerator on a one-sided cell (codebook use)', () => {
    // E[X·1{X∈[lo,1]}] = ∫_lo^1 x f(x) dx for d=8; cross-checked by symmetry:
    // ∫_{-1}^{1} x f(x) dx = 0, so ∫_lo^1 x f = -∫_{-1}^{lo} x f.
    const d = 8;
    const lo = 0.2;
    const right = adaptiveSimpson((x) => x * coordPdf(x, d), lo, 1.0, 1e-12);
    const left = adaptiveSimpson((x) => x * coordPdf(x, d), -1.0, lo, 1e-12);
    expect(Math.abs(right + left)).toBeLessThan(1e-9);
    expect(right).toBeGreaterThan(0);
  });

  it('the spike guard catches a narrow Gaussian spike a coarse grid would miss', () => {
    // A spike of width ~1e-3 centered at 0 over [-1,1]: total mass ≈ 1.
    const s = 1e-3;
    const norm = 1 / (s * Math.sqrt(2 * Math.PI));
    const f = (x: number) => norm * Math.exp(-(x * x) / (2 * s * s));
    const v = adaptiveSimpson(f, -1, 1, 1e-10);
    expect(Math.abs(v - 1)).toBeLessThan(1e-6);
  });
});
