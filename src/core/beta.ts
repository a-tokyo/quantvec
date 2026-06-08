// Beta-distribution math for quantvec's data-oblivious codebooks.
//
// Why this exists: after a random orthonormal rotation, each coordinate of a
// random unit vector in R^d follows a symmetric distribution on [-1, 1] with
// density ∝ (1 - x^2)^((d-3)/2) (TurboQuant, arXiv:2504.19874; see
// docs/research/turboquant.md). Substituting x = 2t - 1 maps this to the
// symmetric Beta(a, a) distribution on [0, 1] with a = (d - 1) / 2. The
// Lloyd-Max codebook wave needs the Beta pdf/cdf/quantile and a numerical
// integrator (for conditional means), all derived here from the math — no
// external statistics library.
//
// Standard references for the algorithms (implemented clean-room from their
// published forms): Lanczos approximation for log-gamma; the regularized
// incomplete beta via Lentz's modified continued fraction ("betai", Numerical
// Recipes §6.4); inverse CDF by bracketed bisection with optional Newton
// polishing.
//
// The numerical integrator used for conditional-mean integrals lives in
// `./integrate` (so the codebook wave can import it directly); we re-export it
// here for convenience and for existing call sites.

export { adaptiveSimpson, IntegrationError } from './integrate';

/** Discriminated, code-tagged error for the beta module. */
export class BetaError extends Error {
  readonly code: 'INVALID_X' | 'INVALID_P' | 'INVALID_PARAM' | 'INVALID_DIM';
  constructor(code: BetaError['code'], message: string) {
    super(message);
    this.name = 'BetaError';
    this.code = code;
  }
}

// ---------------------------------------------------------------------------
// Log-gamma (Lanczos approximation, g = 7, n = 9 coefficients).
// ---------------------------------------------------------------------------

const LANCZOS_G = 7;
const LANCZOS_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/**
 * Natural log of the gamma function for x > 0.
 *
 * Uses the reflection formula for x < 0.5 to stay accurate near the origin.
 */
export function lgamma(x: number): number {
  if (Number.isNaN(x)) {
    throw new BetaError('INVALID_PARAM', `lgamma: x must be a number, got NaN`);
  }
  if (x < 0.5) {
    // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_COEF[0]!;
  const t = z + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_COEF.length; i++) {
    a += LANCZOS_COEF[i]! / (z + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Log of the Beta function: ln B(a, b) = lnΓ(a) + lnΓ(b) - lnΓ(a+b). */
function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

// ---------------------------------------------------------------------------
// Regularized incomplete beta I_x(a, b) via Lentz's continued fraction.
// ---------------------------------------------------------------------------

const CF_MAX_ITER = 300;
const CF_EPS = 3e-16;
const CF_TINY = 1e-300;

/** Continued-fraction core for I_x(a,b); valid where x < (a+1)/(a+b+2). */
function betacf(x: number, a: number, b: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < CF_TINY) d = CF_TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= CF_MAX_ITER; m++) {
    const m2 = 2 * m;
    // Even step.
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < CF_TINY) d = CF_TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < CF_TINY) c = CF_TINY;
    d = 1 / d;
    h *= d * c;
    // Odd step.
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < CF_TINY) d = CF_TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < CF_TINY) c = CF_TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < CF_EPS) break;
  }
  return h;
}

/**
 * Regularized incomplete beta function I_x(a, b) = betaCdf(x, a, b).
 *
 * @param x parameter in [0, 1]
 * @param a shape parameter, a > 0
 * @param b shape parameter, b > 0
 */
export function betaCdf(x: number, a: number, b: number): number {
  if (!(a > 0) || !(b > 0)) {
    throw new BetaError('INVALID_PARAM', `betaCdf: a, b must be > 0, got a=${a}, b=${b}`);
  }
  if (Number.isNaN(x) || x < 0 || x > 1) {
    throw new BetaError('INVALID_X', `betaCdf: x must be in [0, 1], got ${x}`);
  }
  if (x === 0) return 0;
  if (x === 1) return 1;
  // Front factor x^a (1-x)^b / B(a,b), computed in log space.
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta(a, b));
  // Use the continued fraction in its region of fast convergence, else reflect.
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betacf(x, a, b)) / a;
  }
  return 1 - (front * betacf(1 - x, b, a)) / b;
}

// ---------------------------------------------------------------------------
// Beta pdf.
// ---------------------------------------------------------------------------

/** Beta(a, b) probability density at x ∈ [0, 1]. */
export function betaPdf(x: number, a: number, b: number): number {
  if (!(a > 0) || !(b > 0)) {
    throw new BetaError('INVALID_PARAM', `betaPdf: a, b must be > 0, got a=${a}, b=${b}`);
  }
  if (Number.isNaN(x) || x < 0 || x > 1) {
    throw new BetaError('INVALID_X', `betaPdf: x must be in [0, 1], got ${x}`);
  }
  // Handle endpoints carefully to avoid log(0) when the exponent is 0.
  if (x === 0) {
    if (a < 1) return Infinity;
    if (a === 1) return Math.exp(-lbeta(a, b));
    return 0;
  }
  if (x === 1) {
    if (b < 1) return Infinity;
    if (b === 1) return Math.exp(-lbeta(a, b));
    return 0;
  }
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
}

// ---------------------------------------------------------------------------
// Inverse CDF (quantile) via bisection with Newton polishing.
// ---------------------------------------------------------------------------

/** Inverse of betaCdf: the x ∈ [0, 1] with betaCdf(x, a, b) = p. */
export function betaQuantile(p: number, a: number, b: number): number {
  if (!(a > 0) || !(b > 0)) {
    throw new BetaError('INVALID_PARAM', `betaQuantile: a, b must be > 0, got a=${a}, b=${b}`);
  }
  if (Number.isNaN(p) || p < 0 || p > 1) {
    throw new BetaError('INVALID_P', `betaQuantile: p must be in [0, 1], got ${p}`);
  }
  if (p === 0) return 0;
  if (p === 1) return 1;

  // Two-stage solve: bisection to a safe bracket (robust, always converges),
  // then Newton polishing for the final digits (quadratic, using the pdf as the
  // derivative of the cdf). Bisecting to ~1e-12 keeps the bracket tight enough
  // for full accuracy even for the awkward a<1 endpoint-divergent shapes while
  // still leaving Newton work to do; if a Newton step escapes the bracket or
  // stalls we fall back to a bisection step, so the result is always at least as
  // accurate as bisection.
  let lo = 0;
  let hi = 1;
  for (let iter = 0; iter < 200; iter++) {
    const mid = 0.5 * (lo + hi);
    const c = betaCdf(mid, a, b);
    if (c < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12) break;
  }
  let x = 0.5 * (lo + hi);

  // Newton polish (a few steps) using pdf as the derivative of cdf.
  for (let iter = 0; iter < 12; iter++) {
    const c = betaCdf(x, a, b);
    const d = betaPdf(x, a, b);
    if (!Number.isFinite(d) || d <= 0) break;
    const step = (c - p) / d;
    let nx = x - step;
    // Keep inside the bracket; fall back to a bisection step if Newton escapes.
    if (!(nx > lo && nx < hi)) nx = 0.5 * (lo + hi);
    if (Math.abs(nx - x) < 1e-15) {
      x = nx;
      break;
    }
    x = nx;
    // Tighten the bracket with the Newton result (keeps the fallback safe).
    if (betaCdf(x, a, b) < p) lo = x;
    else hi = x;
  }
  return x;
}

// ---------------------------------------------------------------------------
// Coordinate helpers on [-1, 1] for dimension d.
// ---------------------------------------------------------------------------
//
// A coordinate X of a random unit vector in R^d (post-rotation) lives on
// [-1, 1] with X = 2T - 1 where T ~ Beta(a, a), a = (d - 1) / 2. Hence:
//   pdf_X(x)  = (1/2) pdf_T((x+1)/2)         (Jacobian dT/dX = 1/2)
//   cdf_X(x)  = cdf_T((x+1)/2)
//   q_X(p)    = 2 * q_T(p) - 1

function coordShapeA(d: number): number {
  if (!Number.isFinite(d) || d < 2) {
    throw new BetaError('INVALID_DIM', `coordinate dimension d must be ≥ 2, got ${d}`);
  }
  return (d - 1) / 2;
}

/** Density of one coordinate of a random unit vector in R^d, at x ∈ [-1, 1]. */
export function coordPdf(x: number, d: number): number {
  const a = coordShapeA(d);
  if (Number.isNaN(x) || x < -1 || x > 1) {
    throw new BetaError('INVALID_X', `coordPdf: x must be in [-1, 1], got ${x}`);
  }
  // (1/2) * betaPdf((x+1)/2, a, a)
  return 0.5 * betaPdf((x + 1) / 2, a, a);
}

/** CDF of one coordinate of a random unit vector in R^d, at x ∈ [-1, 1]. */
export function coordCdf(x: number, d: number): number {
  const a = coordShapeA(d);
  if (Number.isNaN(x) || x < -1 || x > 1) {
    throw new BetaError('INVALID_X', `coordCdf: x must be in [-1, 1], got ${x}`);
  }
  return betaCdf((x + 1) / 2, a, a);
}

/** Quantile (inverse CDF) of one coordinate, returning x ∈ [-1, 1]. */
export function coordQuantile(p: number, d: number): number {
  const a = coordShapeA(d);
  if (Number.isNaN(p) || p < 0 || p > 1) {
    throw new BetaError('INVALID_P', `coordQuantile: p must be in [0, 1], got ${p}`);
  }
  return 2 * betaQuantile(p, a, a) - 1;
}
