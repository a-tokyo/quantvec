// General-purpose numerical integration for quantvec.
//
// Why this exists: the Lloyd-Max codebook wave needs conditional-mean integrals
//   E[X | X ∈ cell] = ∫_cell x·f(x) dx / ∫_cell f(x) dx
// over sub-cells of [-1, 1], where f is the coordinate density of a random unit
// vector in R^d, f(x) ∝ (1 - x^2)^((d-3)/2) (TurboQuant, arXiv:2504.19874). For
// the real workload d is always a multiple of 8 (d ≥ 8), so f is BOUNDED (it
// vanishes at ±1) but, for large d, is a SHARP SPIKE near x = 0. Sub-cells are
// often ONE-SIDED, e.g. [lo, 1]. This module provides a sound, always-terminating
// adaptive Simpson integrator tuned for exactly those integrands.
//
// Algorithm (clean-room, from the standard published form): adaptive Simpson
// with Lyness' local error estimate and Richardson extrapolation. On each
// interval [a, b] we compare the one-panel Simpson estimate S to the sum of the
// two half-panel estimates S_L + S_R; the Richardson model gives the local error
// of S_L + S_R as ≈ (S_L + S_R − S) / 15. We recurse, halving the error budget
// at each level, until every leaf interval meets its share of the tolerance.
// Because the per-level budgets sum to the original `tol`, the global error model
// is |I − Î| ≲ tol (absolute).
//
// Two design choices keep it both sound and robust:
//
//  1. FAIL-FAST on non-finite samples. If f returns a non-finite value at any
//     sample point the integrand is not (boundedly) integrable by this rule, so
//     we throw `IntegrationError('NON_FINITE_INTEGRAND')` immediately instead of
//     looping. Callers must pass bounded integrands; the codebook uses d ≥ 8,
//     which is bounded. (The d = 2 endpoint singularity, f ∝ 1/√(1-x²), never
//     occurs in the real workload and is rejected fast rather than mis-handled.)
//
//  2. SPIKE-AWARE refinement. A coarse 5-point Simpson panel can entirely miss a
//     narrow central spike (all samples land in the tails, so S ≈ S_L+S_R ≈ 0 and
//     the error test is fooled into a false "converged"). We defeat this two ways:
//       (a) a non-trivial INITIAL uniform subdivision so the spike is sampled at
//           all (panel width ≪ spike half-width for every d ≥ 8 — see below); and
//       (b) a midpoint-spike guard that forces at least one further subdivision
//           whenever the midpoint sample magnitude greatly exceeds the average of
//           the endpoint magnitudes, i.e. the panel straddles a peak the error
//           test alone cannot see.

/** Discriminated, code-tagged error for the integrator. */
export class IntegrationError extends Error {
  readonly code: 'NON_FINITE_INTEGRAND';
  constructor(code: IntegrationError['code'], message: string) {
    super(message);
    this.name = 'IntegrationError';
    this.code = code;
  }
}

/** Maximum recursion depth — a hard bound that guarantees termination. */
const MAX_DEPTH = 50;

/**
 * Number of equal panels [a, b] is split into before adaptive refinement.
 *
 * Rationale (the d ≥ 8 workload): the coordinate density for dimension d is a
 * spike about x = 0 whose standard deviation is 1/√d, so its half-width is
 * ~1/√d ≥ 1/√d_max. quantvec caps practical dimensions well under ~16k, and
 * even at d = 16384 the spike half-width is ~0.008. With 64 panels over a
 * full [-1, 1] cell the panel width is 1/32 ≈ 0.031 and over a one-sided
 * [0, 1] cell it is 1/64 ≈ 0.016; combined with the midpoint-spike guard below
 * (which subdivides any panel that straddles a peak) the spike is always sampled
 * and resolved. For non-spiky integrands the extra panels cost only a constant
 * factor and never hurt accuracy.
 */
const INIT_PANELS = 64;

/**
 * Factor by which the midpoint sample magnitude must exceed the mean endpoint
 * magnitude before we force a panel to subdivide regardless of the error test.
 * This is the spike guard: a peak hidden between sample points shows up as a
 * midpoint that towers over the endpoints.
 */
const SPIKE_FACTOR = 4;

/** One-panel composite Simpson estimate of ∫_a^b f given f(a), f(b), f(mid). */
function simpson(a: number, b: number, fa: number, fb: number, fm: number): number {
  return ((b - a) / 6) * (fa + 4 * fm + fb);
}

/** Evaluate f and reject non-finite results fast (see module header, choice 1). */
function evalFinite(f: (x: number) => number, x: number): number {
  const v = f(x);
  if (!Number.isFinite(v)) {
    throw new IntegrationError(
      'NON_FINITE_INTEGRAND',
      `integrand is non-finite (f(${x}) = ${v}); adaptiveSimpson requires a bounded integrand`,
    );
  }
  return v;
}

/**
 * Recursive core. `whole` is the one-panel Simpson estimate over [a, b] and
 * `tol` is this interval's share of the global absolute error budget.
 */
function adaptiveSimpsonRec(
  f: (x: number) => number,
  a: number,
  b: number,
  fa: number,
  fb: number,
  fm: number,
  whole: number,
  tol: number,
  depth: number,
): number {
  const m = 0.5 * (a + b);
  const lm = 0.5 * (a + m);
  const rm = 0.5 * (m + b);
  const flm = evalFinite(f, lm);
  const frm = evalFinite(f, rm);
  const left = simpson(a, m, fa, fm, flm);
  const right = simpson(m, b, fm, fb, frm);
  const diff = left + right - whole;

  // Spike guard: a midpoint that towers over the endpoints means the panel
  // straddles a peak the error estimate may not yet see. Force a subdivision.
  const endpointMag = 0.5 * (Math.abs(fa) + Math.abs(fb));
  const spike = Math.abs(fm) > SPIKE_FACTOR * endpointMag && Math.abs(fm) > 0;

  if (depth <= 0 || (!spike && Math.abs(diff) <= 15 * tol)) {
    // Richardson extrapolation: the leading error term of left+right is diff/15.
    return left + right + diff / 15;
  }
  return (
    adaptiveSimpsonRec(f, a, m, fa, fm, flm, left, tol / 2, depth - 1) +
    adaptiveSimpsonRec(f, m, b, fm, fb, frm, right, tol / 2, depth - 1)
  );
}

/**
 * Adaptive Simpson integration of `f` over [a, b] to absolute tolerance `tol`.
 *
 * Error model: |∫_a^b f − result| ≲ tol for bounded, smooth-enough integrands.
 * The routine ALWAYS terminates (recursion is capped at {@link MAX_DEPTH}).
 *
 * Precondition: `f` must be finite at every point in [a, b]. A non-finite sample
 * throws {@link IntegrationError} with code `'NON_FINITE_INTEGRAND'` rather than
 * looping or silently returning garbage. (The quantvec codebook integrates the
 * d ≥ 8 coordinate density, which is bounded; the d = 2 endpoint singularity is
 * outside the workload and is rejected fast.)
 *
 * @throws {IntegrationError} if the integrand is non-finite at any sample point.
 */
export function adaptiveSimpson(
  f: (x: number) => number,
  a: number,
  b: number,
  tol: number,
): number {
  if (a === b) return 0;
  const panels = INIT_PANELS;
  const h = (b - a) / panels;
  // Split the absolute budget evenly across the initial panels so the per-panel
  // errors sum to ≤ tol (the global error model).
  const panelTol = tol / panels;
  let total = 0;
  let prevB = a;
  let fPrev = evalFinite(f, a);
  for (let p = 0; p < panels; p++) {
    const pa = prevB;
    const pb = p === panels - 1 ? b : a + (p + 1) * h;
    const pm = 0.5 * (pa + pb);
    const fa = fPrev;
    const fb = evalFinite(f, pb);
    const fm = evalFinite(f, pm);
    const whole = simpson(pa, pb, fa, fb, fm);
    total += adaptiveSimpsonRec(f, pa, pb, fa, fb, fm, whole, panelTol, MAX_DEPTH);
    prevB = pb;
    fPrev = fb;
  }
  return total;
}
