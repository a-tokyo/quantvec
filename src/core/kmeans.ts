// Seeded k-means for quantvec's IVF coarse quantizer.
//
// Why this exists: IVF (inverted-file) search partitions the corpus into `k`
// coarse cells and probes only the cells nearest the query — sublinear scan cost
// on large corpora. The cells come from plain k-means: k-means++ seeding (Arthur
// & Vassilvitskii, "k-means++: The Advantages of Careful Seeding", SODA 2007)
// followed by Lloyd iterations (Lloyd, "Least Squares Quantization in PCM",
// IEEE Trans. IT 1982). Everything is driven by the caller's seeded RNG (see
// ./rng), so a given (data, seed) pair yields bit-identical centroids on every
// runtime — same determinism contract as the rotation.
//
// Two affinity modes:
//   spherical=false — plain L2: assign by argmin ‖x − c‖², centroids are means.
//   spherical=true  — directions (cosine/dot corpora): assign by argmax ⟨x, c⟩
//     over unit centroids; after each mean step centroids are renormalized to
//     unit length (a zero-norm mean keeps its previous direction). Callers
//     pre-normalize the rows.

import type { Rng } from './rng';

/** Discriminated, code-tagged error for the k-means module. */
export class KMeansError extends Error {
  readonly code: 'INVALID_K' | 'INVALID_DIM' | 'INVALID_LENGTH';
  constructor(code: KMeansError['code'], message: string) {
    super(message);
    this.name = 'KMeansError';
    this.code = code;
  }
}

/** Options for {@link kmeans}. */
export interface KMeansOptions {
  /** Number of centroids; integer in [2, m]. */
  k: number;
  /** Coordinate dimension of each row. */
  dim: number;
  /** Caller-seeded RNG — init is the only stochastic step. */
  rng: Rng;
  /** Spherical mode (unit-direction rows, dot-product affinity). */
  spherical: boolean;
  /** Lloyd iteration cap (default 25). */
  maxIterations?: number;
}

/** Result of {@link kmeans}. */
export interface KMeansResult {
  /** Row-major k·dim centroids (unit-norm rows when spherical). */
  centroids: Float32Array;
  /** Final assignment per input row, length m. */
  assignments: Int32Array;
  /** Lloyd iterations actually run (≤ maxIterations). */
  iterations: number;
}

/** Squared L2 distance between row `r` of `data` and row `c` of `centroids`. */
function distSq(
  data: Float32Array,
  r: number,
  centroids: Float32Array,
  c: number,
  dim: number,
): number {
  let s = 0;
  const rBase = r * dim;
  const cBase = c * dim;
  for (let i = 0; i < dim; i++) {
    const d = data[rBase + i]! - centroids[cBase + i]!;
    s += d * d;
  }
  return s;
}

/** Dot product between row `r` of `data` and row `c` of `centroids`. */
function rowDot(
  data: Float32Array,
  r: number,
  centroids: Float32Array,
  c: number,
  dim: number,
): number {
  let s = 0;
  const rBase = r * dim;
  const cBase = c * dim;
  for (let i = 0; i < dim; i++) s += data[rBase + i]! * centroids[cBase + i]!;
  return s;
}

/**
 * Nearest centroid of row `r` under the chosen affinity. Spherical ranks by
 * dot product over unit centroids (argmax ⟨x, c⟩ — scale-invariant in x, so
 * rows need not be re-normalized per call); L2 ranks by squared distance.
 * First-best wins ties, keeping the result deterministic.
 */
export function nearestCentroid(
  data: Float32Array,
  r: number,
  centroids: Float32Array,
  k: number,
  dim: number,
  spherical: boolean,
): number {
  let best = 0;
  if (spherical) {
    let bestDot = rowDot(data, r, centroids, 0, dim);
    for (let c = 1; c < k; c++) {
      const d = rowDot(data, r, centroids, c, dim);
      if (d > bestDot) {
        bestDot = d;
        best = c;
      }
    }
  } else {
    let bestD = distSq(data, r, centroids, 0, dim);
    for (let c = 1; c < k; c++) {
      const d = distSq(data, r, centroids, c, dim);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
  }
  return best;
}

/** Normalize centroid row `c` to unit length; a zero-norm row is left unchanged. */
function renormalizeRow(centroids: Float32Array, c: number, dim: number): void {
  const base = c * dim;
  let normSq = 0;
  for (let i = 0; i < dim; i++) normSq += centroids[base + i]! * centroids[base + i]!;
  if (normSq === 0) return; // keep the previous direction
  const inv = 1 / Math.sqrt(normSq);
  for (let i = 0; i < dim; i++) centroids[base + i] = centroids[base + i]! * inv;
}

/**
 * k-means++ seeding: the first centroid is drawn uniformly; each subsequent one
 * is drawn with probability ∝ D²(x), the squared L2 distance to the nearest
 * already-chosen centroid (one cumulative-sum pass + one uniform draw each).
 * When every remaining D² is 0 (k duplicate rows), fall back to a uniform draw —
 * a reachable degenerate input, so guarded and tested.
 */
function seedCentroids(
  data: Float32Array,
  m: number,
  k: number,
  dim: number,
  rng: Rng,
): Float32Array {
  const centroids = new Float32Array(k * dim);
  const d2 = new Float64Array(m);

  const first = Math.min(m - 1, Math.floor(rng.nextFloat() * m));
  centroids.set(data.subarray(first * dim, first * dim + dim), 0);
  for (let r = 0; r < m; r++) d2[r] = distSq(data, r, centroids, 0, dim);

  for (let c = 1; c < k; c++) {
    let total = 0;
    for (let r = 0; r < m; r++) total += d2[r]!;
    let pick: number;
    if (total > 0) {
      // Inverse-CDF draw over the D² weights.
      const target = rng.nextFloat() * total;
      let acc = 0;
      pick = m - 1; // float-rounding fallback: the last row
      for (let r = 0; r < m; r++) {
        acc += d2[r]!;
        if (acc > target) {
          pick = r;
          break;
        }
      }
    } else {
      pick = Math.min(m - 1, Math.floor(rng.nextFloat() * m));
    }
    centroids.set(data.subarray(pick * dim, pick * dim + dim), c * dim);
    // Fold the new centroid into the D² table.
    for (let r = 0; r < m; r++) {
      const d = distSq(data, r, centroids, c, dim);
      if (d < d2[r]!) d2[r] = d;
    }
  }
  return centroids;
}

/**
 * Run seeded k-means++ + Lloyd over `m` row-major rows of `data` (length m·dim).
 *
 * Deterministic for a fixed (data, rng seed): k-means++ seeding is the only
 * stochastic step, assignment ties break first-best, and the loop stops the
 * first iteration that changes zero assignments (exact, float-tolerance-free)
 * or at `maxIterations`. Empty clusters are repaired each round by re-seeding
 * from the row currently farthest from its centroid (first-max wins).
 *
 * @throws {KMeansError} `'INVALID_DIM'` on a non-positive-integer dim;
 *   `'INVALID_LENGTH'` if `data.length !== m·dim`; `'INVALID_K'` unless
 *   `2 ≤ k ≤ m` (integer).
 */
export function kmeans(data: Float32Array, m: number, opts: KMeansOptions): KMeansResult {
  const { k, dim, rng, spherical } = opts;
  const maxIterations = opts.maxIterations ?? 25;

  if (!Number.isInteger(dim) || dim <= 0) {
    throw new KMeansError('INVALID_DIM', `dim must be a positive integer, got ${dim}`);
  }
  if (!Number.isInteger(m) || m <= 0 || data.length !== m * dim) {
    throw new KMeansError(
      'INVALID_LENGTH',
      `data length ${data.length} must equal m·dim = ${m * dim}`,
    );
  }
  if (!Number.isInteger(k) || k < 2 || k > m) {
    throw new KMeansError('INVALID_K', `k must be an integer in [2, m=${m}], got ${k}`);
  }

  const centroids = seedCentroids(data, m, k, dim, rng);
  if (spherical) for (let c = 0; c < k; c++) renormalizeRow(centroids, c, dim);

  const assignments = new Int32Array(m).fill(-1);
  const counts = new Int32Array(k);
  const sums = new Float64Array(k * dim);

  let iterations = 0;
  while (iterations < maxIterations) {
    iterations++;

    // ── Assign ────────────────────────────────────────────────────────────
    let changed = 0;
    for (let r = 0; r < m; r++) {
      const c = nearestCentroid(data, r, centroids, k, dim, spherical);
      if (c !== assignments[r]) {
        assignments[r] = c;
        changed++;
      }
    }

    // ── Update means (f64 accumulation) ───────────────────────────────────
    counts.fill(0);
    sums.fill(0);
    for (let r = 0; r < m; r++) {
      const c = assignments[r]!;
      counts[c]!++;
      const rBase = r * dim;
      const cBase = c * dim;
      for (let i = 0; i < dim; i++) sums[cBase + i] += data[rBase + i]!;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue; // repaired below; keep the previous centroid
      const inv = 1 / counts[c]!;
      const base = c * dim;
      for (let i = 0; i < dim; i++) centroids[base + i] = sums[base + i]! * inv;
      if (spherical) renormalizeRow(centroids, c, dim);
    }

    // ── Empty-cluster repair ──────────────────────────────────────────────
    // Re-seed each empty centroid from the row farthest (L2) from its current
    // centroid (first-max wins), claim that row, and let the next assign pass
    // settle the rest. Bounded: each repair fills one empty cluster.
    for (let c = 0; c < k; c++) {
      if (counts[c]! > 0) continue;
      let farRow = 0;
      let farD = -1;
      for (let r = 0; r < m; r++) {
        if (counts[assignments[r]!]! <= 1) continue; // don't orphan a singleton
        const d = distSq(data, r, centroids, assignments[r]!, dim);
        if (d > farD) {
          farD = d;
          farRow = r;
        }
      }
      counts[assignments[farRow]!]!--;
      assignments[farRow] = c;
      counts[c] = 1;
      centroids.set(data.subarray(farRow * dim, farRow * dim + dim), c * dim);
      if (spherical) renormalizeRow(centroids, c, dim);
      changed++;
    }

    if (changed === 0) break;
  }

  return { centroids, assignments, iterations };
}
