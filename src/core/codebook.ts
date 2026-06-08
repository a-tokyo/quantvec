// Lloyd-Max scalar quantizer for quantvec's data-oblivious codebooks.
//
// Why this exists: after the random rotation (see ./rotation), each coordinate of
// a normalized vector follows the universal coordinate density f(x; d) on [-1, 1]
// — a (rescaled) Beta((d-1)/2,(d-1)/2) (see ./beta). TurboQuant (Algorithm 1,
// arXiv:2504.19874; docs/research/turboquant.md) quantizes each coordinate with
// the MSE-optimal scalar quantizer for THAT density, computed offline from the
// distribution alone. Because the density depends only on the dimension d (not on
// any data), the codebook is fully determined by (dim, bits): no training, no RNG.
//
// MSE-optimal scalar quantization is the Lloyd-Max problem. Its fixed-point
// (necessary) conditions for a density f with n cells [t_{i-1}, t_i] are:
//
//   centroid   c_i = E[X | X ∈ cell_i] = ∫_{t_{i-1}}^{t_i} x·f dx / ∫_{t_{i-1}}^{t_i} f dx
//   boundary   t_i = (c_i + c_{i+1}) / 2        (nearest-centroid decision)
//   with the fixed support endpoints t_0 = -1, t_n = 1.
//
// We iterate these two conditions (the Lloyd-Max algorithm) to convergence,
// integrating the conditional-mean numerator/denominator with the shared adaptive
// Simpson integrator (see ./integrate). Initialization at the per-cell median
// quantiles coordQuantile((i+0.5)/n, d) is symmetric and well inside each cell, so
// the iteration converges monotonically (Lloyd's algorithm never increases
// distortion) to the unique symmetric optimum for these log-concave Beta densities.
//
// The codebook is validated against an independent scipy reference and against
// TurboQuant's Theorem-1 distortion envelope (see codebook.test.ts).

import { coordPdf, coordQuantile } from './beta';
import { adaptiveSimpson } from './integrate';

/** Discriminated, code-tagged error for the codebook module. */
export class CodebookError extends Error {
  readonly code: 'INVALID_DIM' | 'INVALID_BITS';
  constructor(code: CodebookError['code'], message: string) {
    super(message);
    this.name = 'CodebookError';
    this.code = code;
  }
}

/** A scalar quantizer: cell boundaries (length n+1) and centroids (length n). */
export interface Codebook {
  /** Decision boundaries t_0..t_n on [-1, 1]; t_0 = -1, t_n = 1, strictly increasing. */
  boundaries: Float32Array;
  /** Reconstruction levels c_0..c_{n-1}, strictly increasing, with c_i ∈ (t_i, t_{i+1}). */
  centroids: Float32Array;
}

/** Allowed quantizer bit-widths. */
export type Bits = 2 | 3 | 4;

/** Absolute integration tolerance for the conditional-mean integrals. */
const INTEGRATION_TOL = 1e-10;

/** Lloyd-Max convergence threshold (max centroid move) and hard iteration cap. */
const CONVERGENCE_TOL = 1e-12;
const MAX_ITERS = 1000;

/** Reject dims that are not a positive multiple of 8. */
function validateDim(dim: number): void {
  if (!Number.isInteger(dim) || dim <= 0 || dim % 8 !== 0) {
    throw new CodebookError('INVALID_DIM', `dim must be a positive multiple of 8, got ${dim}`);
  }
}

/** Reject bit-widths outside {2, 3, 4}. */
function validateBits(bits: number): asserts bits is Bits {
  if (bits !== 2 && bits !== 3 && bits !== 4) {
    throw new CodebookError('INVALID_BITS', `bits must be one of {2, 3, 4}, got ${bits}`);
  }
}

/**
 * Build the MSE-optimal Lloyd-Max scalar codebook for the coordinate density of
 * dimension `dim` at `bits` bits per coordinate (n = 2^bits levels).
 *
 * Deterministic in (dim, bits): no RNG. The result is symmetric about 0 because
 * the coordinate density is symmetric.
 *
 * @throws {CodebookError} `'INVALID_DIM'` / `'INVALID_BITS'` on bad arguments.
 */
export function buildCodebook(dim: number, bits: Bits): Codebook {
  validateDim(dim);
  validateBits(bits);

  const n = 1 << bits;
  const f = (x: number): number => coordPdf(x, dim);

  // Initialize centroids at the per-cell median quantiles: coordQuantile maps the
  // probability mass (i+0.5)/n to its x value, giving a symmetric, well-separated
  // start strictly inside each cell.
  const centroids = new Float64Array(n);
  for (let i = 0; i < n; i++) centroids[i] = coordQuantile((i + 0.5) / n, dim);

  const boundaries = new Float64Array(n + 1);
  boundaries[0] = -1;
  boundaries[n] = 1;

  // Lloyd-Max iteration: alternately recompute boundaries (centroid midpoints)
  // and centroids (conditional means), until the largest centroid move is below
  // the convergence tolerance or the iteration cap is hit (guaranteed termination).
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    // Boundaries = midpoints of adjacent centroids; endpoints fixed at ±1.
    for (let i = 1; i < n; i++) boundaries[i] = 0.5 * (centroids[i - 1]! + centroids[i]!);

    // Centroids = conditional means over their cells.
    let maxMove = 0;
    for (let i = 0; i < n; i++) {
      const lo = boundaries[i]!;
      const hi = boundaries[i + 1]!;
      const denom = adaptiveSimpson(f, lo, hi, INTEGRATION_TOL);
      // Empty cell (no mass) — should not happen for these continuous densities,
      // but guard so a degenerate cell keeps its centroid rather than dividing by 0.
      if (denom <= 0) continue;
      const numer = adaptiveSimpson((x) => x * f(x), lo, hi, INTEGRATION_TOL);
      const next = numer / denom;
      const move = Math.abs(next - centroids[i]!);
      if (move > maxMove) maxMove = move;
      centroids[i] = next;
    }

    if (maxMove < CONVERGENCE_TOL) break;
  }

  // Recompute boundaries one final time from the converged centroids so the
  // returned boundaries are exactly the midpoints of the returned centroids.
  for (let i = 1; i < n; i++) boundaries[i] = 0.5 * (centroids[i - 1]! + centroids[i]!);

  // Narrow to Float32 for storage / hot-path use.
  const cOut = new Float32Array(n);
  for (let i = 0; i < n; i++) cOut[i] = centroids[i]!;
  const bOut = new Float32Array(n + 1);
  for (let i = 0; i <= n; i++) bOut[i] = boundaries[i]!;

  return { boundaries: bOut, centroids: cOut };
}

/**
 * Map a coordinate value to its cell index in [0, n-1] via binary search over the
 * (sorted) interior boundaries.
 *
 * The boundaries are the Lloyd-Max decision thresholds (centroid midpoints), so
 * the returned cell is the nearest-centroid cell. Inputs outside [-1, 1] are
 * clamped into the two extreme cells. `boundaries` must be the array from
 * {@link buildCodebook} (length n+1, strictly increasing).
 */
export function quantizeCoord(x: number, boundaries: Float32Array): number {
  const n = boundaries.length - 1;
  // Binary search for the rightmost interior boundary < x. Interior boundaries
  // are indices 1..n-1; cell i is [boundaries[i], boundaries[i+1]).
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1; // upper mid to converge toward the right cell
    if (x >= boundaries[mid]!) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * MSE distortion of a codebook against its density:
 *   D = Σ_i ∫_{cell_i} (x − c_i)² · f(x) dx.
 *
 * Used by tests to compare against TurboQuant's Theorem-1 envelope and to confirm
 * the ~4× per-bit decrease. `codebook` must be the output of {@link buildCodebook}
 * for the same (dim, bits).
 *
 * @throws {CodebookError} `'INVALID_DIM'` / `'INVALID_BITS'` on bad arguments.
 */
export function mseDistortion(dim: number, bits: Bits, codebook: Codebook): number {
  validateDim(dim);
  validateBits(bits);
  const { boundaries, centroids } = codebook;
  const n = 1 << bits;
  const f = (x: number): number => coordPdf(x, dim);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const c = centroids[i]!;
    const lo = boundaries[i]!;
    const hi = boundaries[i + 1]!;
    total += adaptiveSimpson((x) => (x - c) * (x - c) * f(x), lo, hi, INTEGRATION_TOL);
  }
  return total;
}
