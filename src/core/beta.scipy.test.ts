// Clean-room validation oracle: cross-check our Beta implementation against
// scipy.stats.beta — an independent reference, not derived from our code.
//
// The reference values below were generated with scipy 1.13.1:
//
//   from scipy.stats import beta
//   for (x,a,b) in CDF_PTS: beta.cdf(x, a, b)
//   for (p,a,b) in PPF_PTS: beta.ppf(p, a, b)
//
// They are embedded as constants so the test runs in any isomorphic runtime
// without a Python dependency (the core stays node-free). Regenerate with the
// snippet above if the reference set changes.

import { describe, expect, it } from 'vitest';
import { betaCdf, betaQuantile } from './beta';

/** scipy.stats.beta.cdf(x, a, b) reference values (scipy 1.13.1). */
const CDF_REF: Array<{ x: number; a: number; b: number; value: number }> = [
  { x: 0.1, a: 2, b: 3, value: 0.05230000000000001 },
  { x: 0.5, a: 2, b: 3, value: 0.6875 },
  { x: 0.9, a: 2, b: 3, value: 0.9963 },
  { x: 0.3, a: 0.5, b: 0.5, value: 0.36901011956554536 },
  { x: 0.5, a: 5, b: 2, value: 0.109375 },
  { x: 0.7, a: 10, b: 10, value: 0.967446643118699 },
  { x: 0.5, a: 0.5, b: 0.5, value: 0.5000000000000001 },
];

/** scipy.stats.beta.ppf(p, a, b) reference values (scipy 1.13.1). */
const PPF_REF: Array<{ p: number; a: number; b: number; value: number }> = [
  { p: 0.05, a: 2, b: 3, value: 0.09761146288641434 },
  { p: 0.5, a: 2, b: 3, value: 0.3857275681323895 },
  { p: 0.95, a: 2, b: 3, value: 0.7513953742698181 },
  { p: 0.25, a: 0.5, b: 0.5, value: 0.14644660940672624 },
  { p: 0.5, a: 5, b: 2, value: 0.73555001670434 },
  { p: 0.9, a: 10, b: 10, value: 0.6420701198065422 },
];

describe('beta scipy cross-check (validation oracle)', () => {
  it('betaCdf matches scipy.stats.beta.cdf within 1e-6', () => {
    for (const { x, a, b, value } of CDF_REF) {
      expect(Math.abs(betaCdf(x, a, b) - value)).toBeLessThan(1e-6);
    }
  });

  it('betaQuantile matches scipy.stats.beta.ppf within 1e-6', () => {
    for (const { p, a, b, value } of PPF_REF) {
      expect(Math.abs(betaQuantile(p, a, b) - value)).toBeLessThan(1e-6);
    }
  });
});
